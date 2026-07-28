"""Orchestrate configurable BEV encoding, lane decoding, and supervision."""
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .curve_nms import curve_nms


@MODELS.register_module()
class BEVLaneHead(nn.Module):
    """Small interface over configurable BEV encoder, decoder, and loss modules."""

    def __init__(self, bev_encoder, lane_decoder, loss_module,
                 anchor_prior=None, proposal_expander=None,
                 query_sampler=None, test_cfg=None, **kwargs):
        super().__init__()
        self.bev_encoder = MODELS.build(bev_encoder)
        self.lane_decoder = MODELS.build(lane_decoder)
        self.loss_module = MODELS.build(loss_module)
        self.anchor_prior = (
            MODELS.build(anchor_prior) if anchor_prior is not None else None)
        self.proposal_expander = (
            MODELS.build(proposal_expander)
            if proposal_expander is not None else None)
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
                 query_feature_sampler=None, required_query_mask=None,
                 required_query_selector=None):
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
            query_feature_sampler=query_feature_sampler,
            required_query_mask=required_query_mask,
            required_query_selector=required_query_selector)
        if anchor_prior is not None:
            selected = predictions.get('selected_query_indices')
            for key in (
                    'anchor_points', 'shape_templates', 'shape_points',
                    'shape_lateral_slopes',
                    'shape_vertical_slopes', 'shape_control_yaw_degrees',
                    'shape_control_pitch_degrees',
                    'selected_template_indices', 'selected_seed_indices'):
                if key in anchor_prior:
                    value = anchor_prior[key]
                    if (selected is not None and hasattr(value, 'shape')
                            and value.ndim >= 2
                            and value.shape[:2] == (
                                selected.size(0),
                                self.lane_decoder.num_query)):
                        index = selected
                        for _ in range(value.ndim - 2):
                            index = index.unsqueeze(-1)
                        value = value.gather(
                            1, index.expand(
                                *selected.shape, *value.shape[2:]))
                    predictions[key] = value
            if 'unique_template_mask' in anchor_prior:
                predictions['unique_template_counts'] = (
                    anchor_prior['unique_template_mask'].sum(dim=-1))
        return predictions

    def _anchor_prior(self, img_feats, pts_feats, img_metas):
        if self.anchor_prior is None:
            return None
        prior = self.anchor_prior(pts_feats, img_feats, img_metas)
        if self.proposal_expander is not None:
            prior.update(self.proposal_expander(
                prior, img_feats, img_metas))
        return prior

    @staticmethod
    def _lidar_occupancy(pts_feats, lidar_bev):
        if (not isinstance(pts_feats, dict)
                or pts_feats.get('occupancy') is None
                or lidar_bev is None):
            return None
        return F.interpolate(
            pts_feats['occupancy'].float(),
            size=lidar_bev.shape[-2:], mode='nearest') > .5

    def _query_feature_sampler(self, img_feats, lidar_bev, img_metas,
                               lidar_occupancy=None):
        if self.query_sampler is None:
            return None

        def sample(anchor_points, query_features):
            return self.query_sampler(
                anchor_points, query_features, img_feats, lidar_bev,
                img_metas, lidar_occupancy=lidar_occupancy)

        return sample

    def loss(self, img_feats, pts_feats, img_metas, prev_bev=None, **targets):
        anchor_prior = self._anchor_prior(img_feats, pts_feats, img_metas)
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        lidar_occupancy = self._lidar_occupancy(pts_feats, current_bev)
        prior_lanes = targets['gt_lanes_3d']
        prior_visibility = targets['gt_lanes_visibility']
        if (targets.get('gt_lanes_3d_enhanced') is not None
                and targets.get('gt_lanes_visibility_enhanced') is not None):
            prior_lanes = targets['gt_lanes_3d_enhanced']
            prior_visibility = targets['gt_lanes_visibility_enhanced']
        required_query_selector = None
        if hasattr(self.loss_module, 'required_prediction_mask'):
            def required_query_selector(predictions):
                return self.loss_module.required_prediction_mask(
                    predictions, prior_lanes, prior_visibility)
        losses = self.loss_module(self._predict(
            current_bev, anchor_prior=anchor_prior,
            query_feature_sampler=self._query_feature_sampler(
                img_feats, current_bev, img_metas, lidar_occupancy),
            required_query_selector=required_query_selector), **targets)
        if anchor_prior is not None:
            losses.update(self.anchor_prior.loss(
                anchor_prior, prior_lanes, prior_visibility))
            if self.proposal_expander is not None:
                losses.update(self.proposal_expander.loss(
                    anchor_prior, prior_lanes, prior_visibility))
        return losses

    def predict(self, img_feats, pts_feats, img_metas, prev_bev=None, **kwargs):
        anchor_prior = self._anchor_prior(img_feats, pts_feats, img_metas)
        current_bev = self.forward_bev(img_feats, pts_feats, img_metas)
        lidar_occupancy = self._lidar_occupancy(pts_feats, current_bev)
        predictions = self._predict(
            current_bev, anchor_prior=anchor_prior,
            query_feature_sampler=self._query_feature_sampler(
                img_feats, current_bev, img_metas, lidar_occupancy))
        category_probability = predictions['pred_category'].softmax(dim=-1)
        _, category = category_probability.max(dim=-1)
        scores = predictions['pred_cls'].sigmoid()
        visibility = predictions['pred_vis'].sigmoid()
        visibility_threshold = float(
            (self.curve_nms_cfg or {}).get('visibility_threshold', .5))
        candidate_queries = (
            (scores >= self.score_threshold)
            & (visibility >= visibility_threshold).any(dim=-1))
        valid_queries = candidate_queries
        nms_keep_queries = scores.new_ones(scores.shape, dtype=bool)
        if self.curve_nms_cfg is not None:
            nms_keep_queries = curve_nms(
                predictions['pred_x'], predictions['pred_z'], visibility,
                scores,
                tube_radius=float(
                    self.curve_nms_cfg.get('tube_radius', 0.5)),
                z_weight=float(
                    self.curve_nms_cfg.get('z_weight', 1.0)),
                epsilon=float(
                    self.curve_nms_cfg.get('epsilon', 1e-6)),
                iou_threshold=float(
                    self.curve_nms_cfg.get('iou_threshold', 0.6)),
                containment_threshold=float(
                    self.curve_nms_cfg.get(
                        'containment_threshold', 0.9)),
                candidate_mask=candidate_queries)
            valid_queries &= nms_keep_queries
        output = dict(
            x=predictions['pred_x'], z=predictions['pred_z'],
            visibility=visibility, scores=scores,
            category=category,
            valid_queries=valid_queries,
            nms_keep_queries=nms_keep_queries,
            anchor_y_steps=self.anchor_y_steps)
        if anchor_prior is not None:
            for key in (
                    'anchor_points', 'shape_templates', 'shape_points',
                    'shape_lateral_slopes', 'shape_vertical_slopes',
                    'shape_control_yaw_degrees',
                    'shape_control_pitch_degrees',
                    'unique_template_counts',
                    'selected_template_indices', 'selected_seed_indices'):
                if key in predictions:
                    output[key] = predictions[key]
        return output
