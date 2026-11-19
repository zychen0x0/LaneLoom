"""Visible-coverage supervision uses observed ground truth."""
import torch
from torch import nn

from LaneLoom.models.head.laneloom_bev_lane_head import BEVLaneHead


def test_q0_without_full_targets_receives_observed_gt_and_visibility():
    class DecoderLoss(nn.Module):
        def forward(self, predictions, **targets):
            return {}

    class Prior(nn.Module):
        def loss(self, prior, lanes, visibility, **kwargs):
            return {}

    class Expander(nn.Module):
        def loss(self, prior, lanes, visibility, **kwargs):
            self.lanes, self.visibility = lanes, visibility
            return {}

    head = BEVLaneHead.__new__(BEVLaneHead)
    nn.Module.__init__(head)
    head.loss_module, head.anchor_prior, head.proposal_expander = DecoderLoss(), Prior(), Expander()
    head.use_enhanced_prior_gt = False
    head.q0_raw_visible_supervision = False
    head._anchor_prior = lambda *args: {}
    head.forward_bev = lambda *args: None
    head._lidar_occupancy = lambda *args: None
    head._query_feature_sampler = lambda *args: None
    head._predict = lambda *args, **kwargs: {}
    lanes = torch.zeros(1, 1, 20, 3)
    visibility = torch.zeros(1, 1, 20)
    visibility[..., 2:10] = 1
    head.loss(None, None, [{}], gt_lanes_3d=lanes,
              gt_lanes_visibility=visibility,
              gt_lanes_3d_enhanced=torch.ones_like(lanes),
              gt_lanes_visibility_enhanced=torch.ones_like(visibility))
    assert head.proposal_expander.lanes is lanes
    assert head.proposal_expander.visibility is visibility
