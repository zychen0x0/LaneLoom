"""The coverage experiment must not silently reintroduce extrapolated GT."""
from pathlib import Path

from mmengine.config import Config
import torch
from torch import nn

from LaneLoom.models.head.laneloom_bev_lane_head import BEVLaneHead


def test_coverage_configs_keep_the_observed_gt_contract():
    root = Path(__file__).resolve().parents[2]
    for name, budget in (('laneloom_openlane_q0_visible_coverage.py', 64),
                         ('laneloom_openlane_q0_visible_coverage_no_prune.py', None)):
        cfg = Config.fromfile(root / 'configs/ablations' / name)
        assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_template_targets is False
        assert cfg.model.lane_head.anchor_prior.shape_supervision == 'visible_modes'
        assert cfg.model.lane_head.lane_decoder.layer_query_budget == budget
        assert cfg.model.lane_head.lane_decoder.num_query == 505
        assert cfg.model.lane_head.anchor_prior.max_templates == 5
        assert cfg.model.lane_head.proposal_expander.num_x_seeds == 101
        assert cfg.model.lane_head.test_cfg.score_threshold == .5
        assert cfg.model.lane_head.use_enhanced_prior_gt is False
        assert cfg.val_evaluator.use_temporal_gt_supplement is False


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
