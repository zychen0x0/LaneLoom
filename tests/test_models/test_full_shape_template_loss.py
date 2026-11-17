import torch
import torch.nn as nn

from LaneLoom.detectors.laneloom import LaneLoom
from LaneLoom.models.head.laneloom_bev_lane_head import BEVLaneHead
from LaneLoom.models.head.lidar_shape_token_prior import LiDARShapeTokenPrior
from LaneLoom.models.head.shape_proposal_expander import ShapeProposalExpander
from LaneLoom.models.losses.laneloom_topk_loss import LaneLossBase


ANCHORS = [2.0 * index for index in range(1, 51)]


def _prior():
    return LiDARShapeTokenPrior(
        lidar_in_channels=4,
        anchor_y_steps=ANCHORS,
        max_templates=5,
        hidden_channels=8,
        supervise_shapes=True,
        shape_supervision='full_templates',
    )


def test_full_template_supervision_matches_valid_targets_one_to_one():
    prior = _prior()
    y = torch.tensor(ANCHORS)
    straight = torch.stack((0.01 * y, y, 0.02 * y), dim=-1)
    turn = torch.stack((0.002 * y.square(), y, -0.01 * y), dim=-1)
    targets = torch.zeros(1, 5, len(ANCHORS), 3)
    targets[:, :, :, 1] = y
    targets[0, 0] = straight
    targets[0, 1] = turn
    valid = torch.tensor([[True, True, False, False, False]])
    predicted = targets.clone()

    losses = prior.loss(
        dict(shape_points=predicted, from_lidar=True),
        lanes_3d=torch.empty(1, 0, len(ANCHORS), 3),
        visibility=torch.empty(1, 0, len(ANCHORS)),
        shape_templates=targets,
        shape_template_valid=valid,
    )

    assert losses['loss_lidar_shape_template'].item() < 1e-7
    assert losses['diag_gt_shape_template_count'].item() == 2


def test_full_template_supervision_uses_far_range_anchors():
    prior = _prior()
    y = torch.tensor(ANCHORS)
    target = torch.stack((0.01 * y, y, 0.02 * y), dim=-1)
    targets = target.view(1, 1, len(ANCHORS), 3).expand(1, 5, -1, -1).clone()
    valid = torch.tensor([[True, False, False, False, False]])
    predicted = targets.clone()
    predicted[0, :, -1, 0] += 2.0

    losses = prior.loss(
        dict(shape_points=predicted, from_lidar=True),
        lanes_3d=torch.empty(1, 0, len(ANCHORS), 3),
        visibility=torch.empty(1, 0, len(ANCHORS)),
        shape_templates=targets,
        shape_template_valid=valid,
    )

    assert losses['loss_lidar_shape_template'].item() > 0


def test_template_z_confidence_weakens_only_unsupported_z_error():
    prior = _prior()
    targets = torch.zeros(1, 5, len(ANCHORS), 3)
    valid = torch.tensor([[True, False, False, False, False]])
    predicted = targets.clone()
    predicted[0, :, -1, 2] = 2.0
    weights = torch.zeros(1, 5, len(ANCHORS))
    weights[0, 0, :-1] = 1.0
    weights[0, 0, -1] = 0.1
    common = dict(
        prior=dict(shape_points=predicted, from_lidar=True),
        lanes_3d=torch.empty(1, 0, len(ANCHORS), 3),
        visibility=torch.empty(1, 0, len(ANCHORS)),
        shape_templates=targets,
        shape_template_valid=valid)

    unweighted = prior.loss(**common)['loss_lidar_shape_template']
    weighted = prior.loss(
        **common, shape_template_z_weights=weights)[
            'loss_lidar_shape_template']

    assert 0 < weighted.item() < unweighted.item()


def test_q0_z_confidence_weakens_only_unsupported_z_error():
    expander = ShapeProposalExpander(
        pc_range=[-10, 0, -10, 10, 100, 10],
        anchor_y_steps=[5.0, 10.0, 15.0, 20.0],
        num_x_seeds=2, template_feature_channels=4, embed_dims=8,
        q0_loss_type='separated_pointwise', q0_assignment='per_gt_min')
    lanes = torch.zeros(1, 1, 4, 3)
    lanes[0, 0, :, 1] = torch.tensor([5.0, 10.0, 15.0, 20.0])
    candidate = lanes.clone()
    candidate[0, 0, -1, 0] = 1.0
    candidate[0, 0, -1, 2] = 2.0
    expanded = dict(anchor_points=candidate, from_lidar=True)
    visibility = torch.ones(1, 1, 4)
    weights = torch.tensor([[[1.0, 1.0, 1.0, 0.1]]])

    unweighted = expander.loss(expanded, lanes, visibility)
    weighted = expander.loss(expanded, lanes, visibility,
                             z_weights=weights)

    assert 0 < weighted['loss_q0_geometry'].item() < (
        unweighted['loss_q0_geometry'].item())
    assert weighted['diag_q0_x_objective'].item() == (
        unweighted['diag_q0_x_objective'].item())
    assert weighted['diag_q0_x_objective'].item() > 0


def test_full_template_supervision_handles_a_sample_without_reliable_modes():
    prior = _prior()
    predicted = torch.randn(
        1, 5, len(ANCHORS), 3, requires_grad=True)
    targets = torch.zeros_like(predicted)
    valid = torch.zeros(1, 5, dtype=torch.bool)

    losses = prior.loss(
        dict(shape_points=predicted, from_lidar=True),
        lanes_3d=torch.empty(1, 0, len(ANCHORS), 3),
        visibility=torch.empty(1, 0, len(ANCHORS)),
        shape_templates=targets,
        shape_template_valid=valid,
        shape_template_z_weights=torch.zeros(1, 5, len(ANCHORS)),
    )
    losses['loss_lidar_shape_template'].backward()

    assert losses['loss_lidar_shape_template'].item() == 0
    assert torch.isfinite(predicted.grad).all()


def test_lane_head_uses_full_targets_only_for_prior_and_q0():
    class DecoderLoss(nn.Module):
        def forward(self, predictions, **targets):
            self.visibility = targets['gt_lanes_visibility']
            return {'loss_decoder': predictions['value'].sum() * 0}

    class ShapePrior(nn.Module):
        def loss(self, prior, lanes, visibility, **targets):
            self.templates = targets['shape_templates']
            self.template_valid = targets['shape_template_valid']
            self.z_weights = targets['shape_template_z_weights']
            return {'loss_template': prior['value'].sum() * 0}

    class Expander(nn.Module):
        def loss(self, prior, lanes, visibility, z_weights=None):
            self.lanes = lanes
            self.visibility = visibility
            self.z_weights = z_weights
            return {'loss_q0': prior['value'].sum() * 0}

    head = BEVLaneHead.__new__(BEVLaneHead)
    nn.Module.__init__(head)
    head.loss_module = DecoderLoss()
    head.anchor_prior = ShapePrior()
    head.proposal_expander = Expander()
    head._anchor_prior = lambda *args: {'value': torch.ones(1)}
    head.forward_bev = lambda *args: torch.ones(1)
    head._lidar_occupancy = lambda *args: None
    head._query_feature_sampler = lambda *args, **kwargs: None
    head._predict = lambda *args, **kwargs: {'value': torch.ones(1)}

    official_lanes = torch.zeros(1, 2, 50, 3)
    official_visibility = torch.zeros(1, 2, 50)
    official_visibility[:, :, 10:20] = 1
    full_lanes = torch.randn(1, 2, 50, 3)
    full_valid = torch.tensor([[True, False]])
    templates = torch.randn(1, 5, 50, 3)
    template_valid = torch.tensor([[True, False, False, False, False]])
    lane_z_weights = torch.ones(1, 2, 50)
    template_z_weights = torch.ones(1, 5, 50)
    head.loss(
        img_feats=None,
        pts_feats=None,
        img_metas=[{}],
        gt_lanes_3d=official_lanes,
        gt_lanes_visibility=official_visibility,
        gt_lanes_category=torch.zeros(1, 2, dtype=torch.long),
        gt_lanes_3d_full=full_lanes,
        gt_lanes_full_valid=full_valid,
        gt_shape_templates=templates,
        gt_shape_template_valid=template_valid,
        gt_lanes_full_z_weights=lane_z_weights,
        gt_shape_template_z_weights=template_z_weights,
    )

    assert head.anchor_prior.templates is templates
    assert head.anchor_prior.template_valid is template_valid
    assert head.anchor_prior.z_weights is template_z_weights
    assert head.proposal_expander.lanes is full_lanes
    assert head.proposal_expander.z_weights is lane_z_weights
    assert torch.equal(
        head.proposal_expander.visibility[0, 0], torch.ones(50))
    assert torch.equal(
        head.proposal_expander.visibility[0, 1], torch.zeros(50))
    assert head.loss_module.visibility is official_visibility


def test_detector_forwards_full_shape_targets_to_lane_head():
    batch = {
        'gt_lanes_3d': torch.zeros(1),
        'gt_lanes_3d_full': torch.ones(1),
        'gt_lanes_full_valid': torch.ones(1, dtype=torch.bool),
        'gt_shape_templates': torch.ones(1),
        'gt_shape_template_valid': torch.ones(1, dtype=torch.bool),
        'unrelated': torch.ones(1),
    }

    targets = LaneLoom._lane_targets(batch)

    assert set(targets) == {
        'gt_lanes_3d',
        'gt_lanes_3d_full',
        'gt_lanes_full_valid',
        'gt_shape_templates',
        'gt_shape_template_valid',
    }


def test_detector_forwards_z_confidence_to_lane_head():
    batch = {
        'gt_lanes_full_z_weights': torch.ones(1, 2, 50),
        'gt_shape_template_z_weights': torch.ones(1, 5, 50),
    }

    targets = LaneLoom._lane_targets(batch)

    assert set(targets) == set(batch)


def test_decoder_can_keep_current_frame_official_visibility():
    class RecordingLoss(LaneLossBase):
        def _assign(self, predictions, lanes, visibility):
            self.assigned_lanes = lanes
            self.assigned_visibility = visibility
            return []

        def _prediction_losses(self, *args, **kwargs):
            return {}

    loss = RecordingLoss(use_enhanced_gt=False)
    official_lanes = torch.zeros(1, 1, 2, 3)
    official_visibility = torch.tensor([[[1.0, 0.0]]])
    enhanced_lanes = torch.ones(1, 1, 2, 3)
    enhanced_visibility = torch.ones(1, 1, 2)

    loss(
        predictions={},
        gt_lanes_3d=official_lanes,
        gt_lanes_visibility=official_visibility,
        gt_lanes_category=torch.zeros(1, 1, dtype=torch.long),
        gt_lanes_3d_enhanced=enhanced_lanes,
        gt_lanes_visibility_enhanced=enhanced_visibility,
    )

    assert loss.assigned_lanes is official_lanes
    assert loss.assigned_visibility is official_visibility
