"""Orchestrate configurable BEV encoding, lane decoding, and supervision."""
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from .curve_nms import curve_nms, mean_distance_curve_nms


@MODELS.register_module()
class BEVLaneHead(nn.Module):
    """Small interface over configurable BEV encoder, decoder, and loss modules."""

    def __init__(self, bev_encoder, lane_decoder, loss_module,
                 anchor_prior=None, proposal_expander=None,
                 query_sampler=None, test_cfg=None,
                 use_enhanced_prior_gt=True, q0_raw_visible_supervision=False,
                 q0_lane_field_loss_weight=0., **kwargs):
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
        self.use_enhanced_prior_gt = bool(use_enhanced_prior_gt)
        self.q0_raw_visible_supervision = bool(q0_raw_visible_supervision)
        self.q0_lane_field_loss_weight = float(q0_lane_field_loss_weight)
        if self.q0_lane_field_loss_weight < 0 or (
                self.q0_lane_field_loss_weight and not self.q0_raw_visible_supervision):
            raise ValueError('Lane-field supervision requires raw-visible Q0 supervision')
        if self.q0_raw_visible_supervision and (
                self.anchor_prior is None or self.proposal_expander is None
                or self.anchor_prior.supervise_shapes):
            raise ValueError('Raw-visible Q0 supervision requires proposals and disables template losses')

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
        if (getattr(self, 'use_enhanced_prior_gt', True)
                and targets.get('gt_lanes_3d_enhanced') is not None
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
            if self.q0_raw_visible_supervision:
                # Q0 predictions and decoder sampling are complete before raw
                # annotations enter their training-only supervision path.
                from LaneLoom.datasets.openlane.dataset import openlane_raw_visible_gt
                from ..losses.q0_visible_geometry import raw_visible_geometry_loss
                from ..losses.bev_lane_field import bev_lane_field_loss
                if targets.get('raw_label') is None:
                    raise ValueError('Raw-visible Q0 training requires original JSON labels')
                points, visible = openlane_raw_visible_gt(
                    targets['raw_label'], self.proposal_expander.pc_range)
                device = anchor_prior['curve_x_controls'].device
                points, visible = points.to(device), visible.to(device)
                losses.update(raw_visible_geometry_loss(
                    anchor_prior, points, visible,
                    length=float(self.anchor_prior.anchor_y_steps[-1])))
                if self.q0_lane_field_loss_weight:
                    field_losses = bev_lane_field_loss(
                        anchor_prior, points, visible, self.proposal_expander.pc_range)
                    losses.update({key: value * self.q0_lane_field_loss_weight
                                   if key.startswith('loss_') else value
                                   for key, value in field_losses.items()})
                return losses
            losses.update(self.anchor_prior.loss(
                anchor_prior, prior_lanes, prior_visibility,
                shape_templates=targets.get('gt_shape_templates'),
                shape_template_valid=targets.get(
                    'gt_shape_template_valid'),
                shape_template_z_weights=targets.get(
                    'gt_shape_template_z_weights')))
            if self.proposal_expander is not None:
                q0_lanes = targets.get('gt_lanes_3d_full', prior_lanes)
                q0_visibility = prior_visibility
                full_lane_valid = targets.get('gt_lanes_full_valid')
                if full_lane_valid is not None:
                    q0_visibility = full_lane_valid.unsqueeze(-1).expand(
                        -1, -1, q0_lanes.size(-2)).to(q0_lanes.dtype)
                q0_z_weights = targets.get('gt_lanes_full_z_weights')
                if q0_z_weights is None:
                    losses.update(self.proposal_expander.loss(
                        anchor_prior, q0_lanes, q0_visibility))
                else:
                    losses.update(self.proposal_expander.loss(
                        anchor_prior, q0_lanes, q0_visibility,
                        z_weights=q0_z_weights))
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
            tube_nms_enabled = any(
                name in self.curve_nms_cfg
                for name in ('iou_threshold', 'containment_threshold'))
            if tube_nms_enabled:
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
            mean_distance_threshold = self.curve_nms_cfg.get(
                'mean_distance_threshold')
            if mean_distance_threshold is not None:
                distance_keep_queries = mean_distance_curve_nms(
                    predictions['pred_x'], predictions['pred_z'], visibility,
                    scores,
                    distance_threshold=float(mean_distance_threshold),
                    visibility_threshold=visibility_threshold,
                    candidate_mask=valid_queries)
                nms_keep_queries &= distance_keep_queries
                valid_queries &= distance_keep_queries
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
