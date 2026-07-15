"""Orchestrate configurable BEV encoding, lane decoding, and supervision."""
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerBEVLaneHead(nn.Module):
    """Small interface over configurable BEV encoder, decoder, and loss modules."""

    def __init__(self, bev_encoder, lane_decoder, loss_module,
                 anchor_prior=None, query_sampler=None,
                 test_cfg=None, **kwargs):
        super().__init__()
        self.bev_encoder = MODELS.build(bev_encoder)
        self.lane_decoder = MODELS.build(lane_decoder)
        self.loss_module = MODELS.build(loss_module)
        self.anchor_prior = (
            MODELS.build(anchor_prior) if anchor_prior is not None else None)
        self.query_sampler = (
            MODELS.build(query_sampler) if query_sampler is not None else None)
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

    def _predict(self, current_bev, dn_lanes_3d=None,
                 dn_visibility=None, anchor_prior=None,
                 query_feature_sampler=None):
        predictions = self.lane_decoder(
            current_bev,
            dn_lanes_3d=dn_lanes_3d, dn_visibility=dn_visibility,
            initial_reference_controls=(
                (anchor_prior['curve_x_controls'],
                 anchor_prior['curve_z_controls'])
                if anchor_prior is not None else None),
            query_feature_sampler=query_feature_sampler)
        if anchor_prior is not None:
            predictions['road_curvature'] = anchor_prior['curvature']
            predictions['road_grade'] = anchor_prior['grade']
            predictions['anchor_points'] = anchor_prior['anchor_points']
        return predictions

    def _anchor_prior(self, img_feats, pts_feats, img_metas):
        if self.anchor_prior is None:
            return None
        return self.anchor_prior(pts_feats)

    def _query_feature_sampler(self, img_feats, lidar_bev, img_metas):
        if self.query_sampler is None:
            return None

        def sample(anchor_points, visibility):
            return self.query_sampler(
                anchor_points, img_feats, lidar_bev, img_metas,
                visibility=visibility)

        return sample

    def loss(self, img_feats, pts_feats, img_metas, prev_bev=None, **targets):
        anchor_prior = self._anchor_prior(img_feats, pts_feats, img_metas)
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        dn_lanes_3d = targets.get('gt_lanes_3d_enhanced')
        dn_visibility = targets.get('gt_lanes_visibility_enhanced')
        if dn_lanes_3d is None or dn_visibility is None:
            dn_lanes_3d = targets.get('gt_lanes_3d')
            dn_visibility = targets.get('gt_lanes_visibility')
        losses = self.loss_module(self._predict(
            current_bev, dn_lanes_3d=dn_lanes_3d,
            dn_visibility=dn_visibility, anchor_prior=anchor_prior,
            query_feature_sampler=self._query_feature_sampler(
                img_feats, current_bev, img_metas)), **targets)
        if anchor_prior is not None:
            losses.update(self.anchor_prior.loss(
                anchor_prior, targets['gt_lanes_3d'],
                targets['gt_lanes_visibility']))
        return losses

    def predict(self, img_feats, pts_feats, img_metas, prev_bev=None, **kwargs):
        anchor_prior = self._anchor_prior(img_feats, pts_feats, img_metas)
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        predictions = self._predict(
            current_bev, anchor_prior=anchor_prior,
            query_feature_sampler=self._query_feature_sampler(
                img_feats, current_bev, img_metas))
        class_probability = predictions['pred_logits'].softmax(dim=-1)[..., 1:]
        scores, category = class_probability.max(dim=-1)
        output = dict(
            x=predictions['pred_x'], z=predictions['pred_z'],
            visibility=predictions['pred_vis'].sigmoid(), scores=scores,
            category=category,
            valid_queries=scores >= self.score_threshold,
            anchor_y_steps=self.anchor_y_steps)
        if anchor_prior is not None:
            output.update(
                road_curvature=predictions['road_curvature'],
                road_grade=predictions['road_grade'],
                anchor_points=predictions['anchor_points'])
        return output
