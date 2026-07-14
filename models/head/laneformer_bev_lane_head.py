"""Orchestrate configurable BEV encoding, lane decoding, and supervision."""
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerBEVLaneHead(nn.Module):
    """Small interface over configurable BEV encoder, decoder, and loss modules."""

    def __init__(self, bev_encoder, lane_decoder, loss_module,
                 test_cfg=None, **kwargs):
        super().__init__()
        self.bev_encoder = MODELS.build(bev_encoder)
        self.lane_decoder = MODELS.build(lane_decoder)
        self.loss_module = MODELS.build(loss_module)
        self.test_cfg = test_cfg or {}
        self.score_threshold = float(self.test_cfg.get('score_threshold', 0.3))

    @property
    def anchor_y_steps(self):
        return self.lane_decoder.anchor_y_steps

    @property
    def num_anchors(self):
        return self.lane_decoder.num_anchors

    @property
    def num_query(self):
        return self.lane_decoder.num_query

    def forward_bev(self, img_feats, pts_feats, img_metas=None, prev_bev=None,
                    **kwargs):
        return self.bev_encoder.forward_bev(img_feats, pts_feats, img_metas)

    def forward_image_bev(self, img_feats, img_metas=None):
        """Encode image features without introducing LiDAR into history BEV."""
        return self.bev_encoder.forward_image_bev(img_feats, img_metas)

    def _predict(self, current_bev, prev_bev, dn_lanes_3d=None,
                 dn_visibility=None):
        fused_bev = self.bev_encoder.temporal_self_attn(current_bev, prev_bev)
        return self.lane_decoder(
            fused_bev, self.bev_encoder.bev_pos,
            dn_lanes_3d=dn_lanes_3d, dn_visibility=dn_visibility)

    def loss(self, img_feats, pts_feats, img_metas, prev_bev=None, **targets):
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        dn_lanes_3d = targets.get('gt_lanes_3d_enhanced')
        dn_visibility = targets.get('gt_lanes_visibility_enhanced')
        if dn_lanes_3d is None or dn_visibility is None:
            dn_lanes_3d = targets.get('gt_lanes_3d')
            dn_visibility = targets.get('gt_lanes_visibility')
        return self.loss_module(self._predict(
            current_bev, prev_bev, dn_lanes_3d=dn_lanes_3d,
            dn_visibility=dn_visibility), **targets)

    def predict(self, img_feats, pts_feats, img_metas, prev_bev=None, **kwargs):
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        predictions = self._predict(current_bev, prev_bev)
        scores = predictions['pred_cls'].sigmoid()
        return dict(
            x=predictions['pred_x'], z=predictions['pred_z'],
            visibility=predictions['pred_vis'].sigmoid(), scores=scores,
            category=predictions['pred_category'].argmax(dim=-1),
            valid_queries=scores >= self.score_threshold,
            anchor_y_steps=self.anchor_y_steps)
