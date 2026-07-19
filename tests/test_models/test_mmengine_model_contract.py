import inspect
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

import LaneFormer  # noqa: F401 - register the project model.
from LaneFormer.detectors import LaneFormer as LaneFormerModel


def test_laneformer_is_the_single_mmengine_model_entrypoint():
    assert issubclass(LaneFormerModel, BaseModel)
    assert MODELS.get('LaneFormer') is LaneFormerModel

    parameters = inspect.signature(LaneFormerModel.forward).parameters
    assert 'mode' in parameters
    assert parameters.get('return' + '_loss') is None


def test_laneformer_forward_accepts_ddp_keyword_batches():
    signature = inspect.signature(LaneFormerModel.forward)
    signature.bind(
        None,
        mode='loss',
        img='image-batch',
        img_metas='metadata-batch',
        lidar='point-batch')


def test_config_enables_unused_parameter_detection_for_dynamic_bev_branches():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneformer_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    assert cfg.model_wrapper_cfg.type == 'MMDistributedDataParallel'
    assert cfg.model_wrapper_cfg.find_unused_parameters is True


def test_config_routes_ten_lidar_shapes_into_fifty_decoder_queries():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneformer_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    prior = cfg.model.lane_head.anchor_prior
    router = cfg.model.lane_head.proposal_router
    decoder = cfg.model.lane_head.lane_decoder

    assert prior.max_templates == 10
    assert prior.lateral_slope_scale == 0.1
    assert prior.vertical_slope_scale == 0.03
    assert 'max_yaw_degrees' not in prior
    assert 'max_pitch_degrees' not in prior
    assert router.num_x_seeds == router.proposal_budget == decoder.num_query == 50
    assert prior.pc_range == decoder.pc_range
    assert prior.anchor_y_steps == router.anchor_y_steps == decoder.anchor_y_steps
    assert decoder.num_layers == 2


def test_config_uses_two_level_temporal_fv_and_no_image_bev():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneformer_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)

    assert cfg.model.num_levels == 2
    assert cfg.model.img_backbone.out_indices == (1, 2)
    assert cfg.model.lane_head.query_sampler.type == (
        'LaneFormerDualPlaneTemporalSampler')
    assert 'spatial_cross_attn' not in cfg.model.lane_head.bev_encoder
    assert 'lidar_fusion' not in cfg.model.lane_head.bev_encoder
    assert 'temporal_self_attn' not in cfg.model.lane_head.bev_encoder


def test_config_uses_strict_q0_and_one_to_one_assignment():
    config_path = Path(__file__).resolve().parents[2] / 'configs/laneformer_openlane_image_lidar.py'
    cfg = Config.fromfile(config_path)
    loss = cfg.model.lane_head.loss_module

    assert loss.type == 'LaneFormerOneToOneLaneLoss'
    assert loss.assigner.type == 'LaneFormerHungarianAssigner'
    assert loss.visibility_loss_weight == 1.0
    assert 'visibility_dice_loss_weight' not in loss
    assert 'visibility_boundary_loss_weight' not in loss
    assert cfg.model.lane_head.lane_decoder.get('dn_num_groups', 0) == 0
    router = cfg.model.lane_head.proposal_router
    assert router.q0_distance_threshold == 0.3
    assert router.q0_match_ratio == 0.8
    assert router.q0_geometry_loss_weight == 1.0
    assert cfg.model.lane_head.test_cfg.curve_nms.distance_threshold == 0.3
    assert cfg.model.lane_head.test_cfg.curve_nms.match_ratio_threshold == 0.75


def _bare_laneformer():
    model = LaneFormerModel.__new__(LaneFormerModel)
    nn.Module.__init__(model)
    model.frames = (-1, 0)
    model.num_levels = None
    return model


def test_temporal_image_features_keep_every_frame_for_curve_sampling():
    model = _bare_laneformer()
    model.extract_img_feat = Mock(
        return_value=[torch.randn(6, 1, 4, 2, 2)])
    image_queue = torch.randn(2, 3, 1, 3, 4, 4)

    temporal = model.extract_temporal_img_feat(image_queue)

    assert temporal[0].shape == (2, 3, 1, 4, 2, 2)
    assert model.extract_img_feat.call_args.args[0].shape == (6, 1, 3, 4, 4)


def test_current_lidar_backbone_receives_the_complete_aligned_point_queue():
    model = _bare_laneformer()
    model.extract_temporal_img_feat = Mock(
        return_value=[torch.randn(1, 3, 1, 4, 2, 2)])
    model.extract_pts_feat = Mock(return_value=[torch.randn(1, 4, 2, 2)])
    accumulated = torch.tensor([
        [0., 2., -1., 1., 0., 0.],
        [0., 3., -1., 1., 0., 1.],
        [0., 4., -1., 1., 0., 2.],
    ])
    batch = dict(
        img=torch.randn(1, 3, 1, 3, 4, 4),
        lidar=[accumulated],
        img_metas=[dict(queue={0: {}, 1: {}, 2: {}})])

    encoded = model._encode_batch(batch)

    passed_points = model.extract_pts_feat.call_args.args[0]
    assert passed_points[0].size(0) == 3
    assert encoded[3][0].size(0) == 3
    assert len(encoded) == 5
