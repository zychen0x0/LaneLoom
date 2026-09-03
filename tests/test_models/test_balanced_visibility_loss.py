from pathlib import Path

import torch
from mmengine.config import Config
from mmdet3d.registry import MODELS

from LaneLoom.engine import register_all_modules


register_all_modules()


def _single_lane_inputs():
    probabilities = torch.tensor([[[0.8, 0.2, 0.9, 0.1]]])
    predictions = dict(
        pred_x=torch.zeros(1, 1, 4),
        pred_z=torch.zeros(1, 1, 4),
        pred_vis=torch.logit(probabilities).requires_grad_(),
        pred_vis_start=torch.tensor([[0.5]], requires_grad=True),
        pred_vis_end=torch.tensor([[2.5]], requires_grad=True),
        pred_cls=torch.zeros(1, 1, requires_grad=True),
        pred_category=torch.zeros(1, 1, 2, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 4, 3)
    gt_visibility = torch.tensor([[[1., 1., 0., 0.]]])
    gt_category = torch.zeros(1, 1, dtype=torch.long)
    return predictions, gt_lanes, gt_visibility, gt_category


def test_lane_loss_balances_visibility_groups_and_supervises_boundaries():
    loss_module = MODELS.build(dict(
        type='LaneLoss',
        visibility_loss_type='balanced_bce',
        visibility_loss_weight=1.0,
        visibility_start_loss_weight=0.5,
        visibility_end_loss_weight=0.5,
        score_loss_weight=0.0,
        category_loss_weight=0.0,
        x_loss_weight=0.0,
        z_loss_weight=0.0,
        tube_loss_weight=0.0))
    predictions, lanes, visibility, category = _single_lane_inputs()

    losses = loss_module(
        predictions,
        gt_lanes_3d=lanes,
        gt_lanes_visibility=visibility,
        gt_lanes_category=category)

    # Balanced BCE is 1.0601318. Each one-anchor boundary error contributes
    # SmoothL1(1/3, beta=1/3) = 1/6 before its 0.5 weight.
    assert torch.allclose(
        losses['loss_visibility'], torch.tensor(1.2267984), atol=1e-6)
    losses['loss_visibility'].backward()
    assert predictions['pred_vis'].grad.abs().sum() > 0
    assert predictions['pred_vis_start'].grad.abs().sum() > 0
    assert predictions['pred_vis_end'].grad.abs().sum() > 0


def test_lane_loss_keeps_probability_smooth_l1_as_the_default():
    loss_module = MODELS.build(dict(
        type='LaneLoss',
        visibility_loss_weight=1.0,
        score_loss_weight=0.0,
        category_loss_weight=0.0,
        x_loss_weight=0.0,
        z_loss_weight=0.0,
        tube_loss_weight=0.0))
    predictions, lanes, visibility, category = _single_lane_inputs()

    losses = loss_module(
        predictions,
        gt_lanes_3d=lanes,
        gt_lanes_visibility=visibility,
        gt_lanes_category=category)

    assert torch.allclose(
        losses['loss_visibility'], torch.tensor(0.3125), atol=1e-6)


def test_balanced_visibility_ablation_enables_the_new_loss():
    config_path = (
        Path(__file__).resolve().parents[2]
        / 'configs/ablations/laneloom_openlane_image_lidar_balanced_vis_loss.py'
    )

    cfg = Config.fromfile(config_path)
    loss = cfg.model.lane_head.loss_module

    assert cfg.work_dir == (
        './work_dirs/laneloom_openlane_image_lidar_balanced_vis_loss')
    assert loss.visibility_loss_type == 'balanced_bce'
    assert loss.visibility_loss_weight == 1.0
    assert loss.visibility_start_loss_weight == 0.5
    assert loss.visibility_end_loss_weight == 0.5
