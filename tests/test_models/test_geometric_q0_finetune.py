"""Preserve learned initialization while rebuilding the decoder sampling grid."""
import pytest
import torch
from torch import nn

from LaneLoom.detectors.laneloom import LaneLoom
from LaneLoom.models.losses.q0_visible_geometry import raw_visible_geometry_loss
from LaneLoom.models.losses.bev_lane_field import bev_lane_field_loss
from LaneLoom.tools.prepare_q0_finetune_checkpoint import compose_state


def test_lane_targets_preserve_raw_labels():
    raw = [dict(lane_lines=[])]
    assert LaneLoom._lane_targets(dict(raw_label=raw))['raw_label'] is raw


def test_composed_checkpoint_replaces_encoder_and_head_but_preserves_configured_anchors():
    model = nn.Module()
    model.pts_bev_encoder = nn.Linear(2, 2)
    model.img_backbone = nn.Linear(2, 2)
    model.lane_head = nn.Module()
    model.lane_head.anchor_prior = nn.Linear(2, 2)
    model.lane_head.anchor_prior.register_buffer('anchor_y_steps', torch.arange(50))
    model.lane_head.lane_decoder = nn.Linear(2, 2)
    model.lane_head.lane_decoder.register_buffer('anchor_y_steps', torch.arange(50))
    detector = {k: torch.full_like(v, 3) for k, v in model.state_dict().items()}
    q0 = {k.replace('lane_head.anchor_prior.', 'anchor_prior.'): torch.full_like(v, 7)
          for k, v in model.state_dict().items()
          if k.startswith(('pts_bev_encoder.', 'lane_head.anchor_prior.'))}
    q0['anchor_prior.anchor_y_steps'] = torch.arange(20)
    detector['lane_head.lane_decoder.anchor_y_steps'] = torch.arange(20)
    state, rebuilt = compose_state(model, detector, q0)
    model.load_state_dict(state, strict=True)
    assert torch.all(model.pts_bev_encoder.weight == 7)
    assert torch.all(model.lane_head.anchor_prior.weight == 7)
    assert torch.all(model.img_backbone.weight == 3)
    assert torch.all(model.lane_head.lane_decoder.weight == 3)
    assert torch.equal(model.lane_head.lane_decoder.anchor_y_steps, torch.arange(50))
    assert torch.equal(model.lane_head.anchor_prior.anchor_y_steps, torch.arange(50))
    assert 'lane_head.lane_decoder.anchor_y_steps' in rebuilt
    del q0['anchor_prior.weight']
    with pytest.raises(ValueError, match='absent or incompatible'):
        compose_state(model, detector, q0)


def test_q0_loss_diagnostics_can_be_reduced_by_mmengine_train_step():
    controls = torch.zeros(1, 5, 4, requires_grad=True)
    prediction = dict(curve_x_controls=controls, curve_z_controls=controls,
                      lane_field_logits=torch.zeros(1, 1, 10, 10, requires_grad=True),
                      lane_height_field=torch.zeros(1, 1, 10, 10, requires_grad=True))
    points = torch.tensor([[[[0., 10., 0.], [0., 20., 0.]]]])
    visible = torch.ones(1, 1, 2, dtype=torch.bool)
    losses = raw_visible_geometry_loss(prediction, points, visible)
    losses.update(bev_lane_field_loss(prediction, points, visible, [-25, 0, -5, 25, 100, 5]))
    total, logged = LaneLoom.parse_losses(None, losses)
    assert torch.isfinite(total) and total.requires_grad
    assert logged['diag_q0_raw_gt'] == 1
    assert all(torch.is_floating_point(value) for value in logged.values())
    total.backward()
