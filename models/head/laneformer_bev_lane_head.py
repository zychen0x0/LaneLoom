"""Orchestrate configurable BEV encoding, lane decoding, and supervision."""
import torch.nn as nn

from mmdet3d.registry import MODELS

from .curve_nms import curve_nms


@MODELS.register_module()
class LaneFormerBEVLaneHead(nn.Module):
    """Small interface over configurable BEV encoder, decoder, and loss modules."""

    def __init__(self, bev_encoder, lane_decoder, loss_module,
                 anchor_prior=None, proposal_router=None, query_sampler=None,
                 test_cfg=None, **kwargs):
        super().__init__()
        self.bev_encoder = MODELS.build(bev_encoder)
        self.lane_decoder = MODELS.build(lane_decoder)
        self.loss_module = MODELS.build(loss_module)
        self.anchor_prior = (
            MODELS.build(anchor_prior) if anchor_prior is not None else None)
        self.proposal_router = (
            MODELS.build(proposal_router)
            if proposal_router is not None else None)
        self.query_sampler = (
            MODELS.build(query_sampler) if query_sampler is not None else None)
        self.test_cfg = test_cfg or {}
        self.score_threshold = float(self.test_cfg.get('score_threshold', 0.3))
        self.curve_nms_cfg = self.test_cfg.get('curve_nms')

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

    def _predict(self, current_bev, anchor_prior=None,
                 query_feature_sampler=None):
        initial_controls = None
        query_features = None
        if anchor_prior is not None and 'curve_x_controls' in anchor_prior:
            initial_controls = (
                anchor_prior['curve_x_controls'],
                anchor_prior['curve_z_controls'])
            query_features = anchor_prior.get('query_features')
        predictions = self.lane_decoder(
            current_bev,
            query_features=query_features,
            initial_reference_controls=initial_controls,
            query_feature_sampler=query_feature_sampler)
        if anchor_prior is not None:
            for key in (
                    'curvature', 'grade', 'anchor_points', 'shape_templates',
                    'shape_points', 'shape_yaw', 'shape_pitch',
                    'routing_scores',
                    'selected_template_indices', 'selected_seed_indices'):
                if key in anchor_prior:
                    predictions[key] = anchor_prior[key]
            if 'unique_template_mask' in anchor_prior:
                predictions['unique_template_counts'] = (
                    anchor_prior['unique_template_mask'].sum(dim=-1))
        return predictions

    def _anchor_prior(self, img_feats, pts_feats, img_metas):
        if self.anchor_prior is None:
            return None
        prior = self.anchor_prior(pts_feats, img_feats, img_metas)
        if self.proposal_router is not None:
            prior.update(self.proposal_router(prior, img_feats, img_metas))
        return prior

    def _query_feature_sampler(self, img_feats, lidar_bev, img_metas):
        if self.query_sampler is None:
            return None

        def sample(anchor_points):
            return self.query_sampler(
                anchor_points, img_feats, lidar_bev, img_metas)

        return sample

    def loss(self, img_feats, pts_feats, img_metas, prev_bev=None, **targets):
        anchor_prior = self._anchor_prior(img_feats, pts_feats, img_metas)
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        losses = self.loss_module(self._predict(
            current_bev, anchor_prior=anchor_prior,
            query_feature_sampler=self._query_feature_sampler(
                img_feats, current_bev, img_metas)), **targets)
        if anchor_prior is not None:
            losses.update(self.anchor_prior.loss(
                anchor_prior, targets['gt_lanes_3d'],
                targets['gt_lanes_visibility']))
            if self.proposal_router is not None:
                losses.update(self.proposal_router.loss(
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
        visibility = predictions['pred_vis'].sigmoid()
        valid_queries = scores >= self.score_threshold
        nms_keep_queries = scores.new_ones(scores.shape, dtype=bool)
        if self.curve_nms_cfg is not None:
            nms_keep_queries = curve_nms(
                predictions['pred_x'], predictions['pred_z'], visibility,
                scores, visibility_threshold=float(
                    self.curve_nms_cfg.get('visibility_threshold', 0.5)),
                distance_threshold=float(
                    self.curve_nms_cfg.get('distance_threshold', 0.3)),
                match_ratio_threshold=float(
                    self.curve_nms_cfg.get('match_ratio_threshold', 0.75)))
            valid_queries &= nms_keep_queries
        output = dict(
            x=predictions['pred_x'], z=predictions['pred_z'],
            visibility=visibility, scores=scores,
            category=category,
            valid_queries=valid_queries,
            nms_keep_queries=nms_keep_queries,
            anchor_y_steps=self.anchor_y_steps)
        if anchor_prior is not None:
            if 'curvature' in predictions:
                output['road_curvature'] = predictions['curvature']
            if 'grade' in predictions:
                output['road_grade'] = predictions['grade']
            for key in (
                    'anchor_points', 'shape_templates', 'shape_points',
                    'routing_scores', 'unique_template_counts',
                    'selected_template_indices', 'selected_seed_indices'):
                if key in predictions:
                    output[key] = predictions[key]
        return output
