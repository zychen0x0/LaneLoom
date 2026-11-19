"""Ablation variants registered on top of the unchanged LaneLoom modules.

Every class subclasses an existing LaneLoom module and changes exactly one
behavior, so each ablation config differs from the full model in one place.
The original source files are not modified.

Registered names (all in the ``mmdet3d`` MODELS registry):

* ``AblationStraightTemplatePrior``: CQG replaced by fixed straight templates.
  Template control points are zero, so every initial query is a straight
  line at its lateral seed. Template features still feed the query features.
* ``AblationCurrentSweepPillarEncoder``: the LiDAR branch keeps only the
  current sweep. Historical images are still sampled by MTFF.
* ``AblationCurrentImageSampler``: MTFF samples only the current image.
  The accumulated multi-sweep LiDAR BEV is unchanged.
* ``AblationMeanTimeSampler``: temporal convolution replaced by a masked
  mean over the valid frames.
"""

import torch
from mmdet3d.registry import MODELS

from LaneLoom.models.backbones.dynamic_pillar_bev import DynamicPillarBEVEncoder
from LaneLoom.models.head.dual_plane_temporal_sampler import (
    DualPlaneTemporalSampler)
from LaneLoom.models.head.lidar_shape_token_prior import LiDARShapeTokenPrior


@MODELS.register_module()
class AblationStraightTemplatePrior(LiDARShapeTokenPrior):
    """Keep the CQG interface but force every template to a straight line."""

    _EXPECTED_OUTPUTS = 8

    def _relative_controls(self, pts_feats, img_feats=None):
        outputs = super()._relative_controls(pts_feats, img_feats)
        if len(outputs) != self._EXPECTED_OUTPUTS:
            raise RuntimeError(
                'LiDARShapeTokenPrior._relative_controls changed its return '
                f'signature (expected {self._EXPECTED_OUTPUTS} items, got '
                f'{len(outputs)}). Update AblationStraightTemplatePrior.')
        (x_controls, z_controls, lateral_slopes, vertical_slopes,
         yaw_degrees, pitch_degrees, tokens, from_lidar) = outputs
        zeros = torch.zeros_like
        return (zeros(x_controls), zeros(z_controls), zeros(lateral_slopes),
                zeros(vertical_slopes), zeros(yaw_degrees),
                zeros(pitch_degrees), tokens, from_lidar)


@MODELS.register_module()
class AblationCurrentSweepPillarEncoder(DynamicPillarBEVEncoder):
    """Drop historical LiDAR sweeps before painting and pillarization."""

    def _current_only(self, points):
        if not self.has_queue_index or self.queue_length <= 1:
            return points
        current = float(self.queue_length - 1)
        column = self.raw_queue_index
        single = torch.is_tensor(points)
        samples = [points] if single else list(points)
        kept = [sample[sample[:, column].round() == current]
                for sample in samples]
        return kept[0] if single else kept

    def forward(self, points, img_feats=None, img_metas=None, images=None):
        return super().forward(
            self._current_only(points), img_feats=img_feats,
            img_metas=img_metas, images=images)


@MODELS.register_module()
class AblationCurrentImageSampler(DualPlaneTemporalSampler):
    """Mask every historical image sample; the current frame is the last."""

    def _sample_image(self, curves, query_features, image_features,
                      img_metas):
        sampled, valid = super()._sample_image(
            curves, query_features, image_features, img_metas)
        if sampled.size(-2) > 1:
            sampled = sampled.clone()
            valid = valid.clone()
            sampled[..., :-1, :] = 0
            valid[..., :-1] = False
        return sampled, valid


@MODELS.register_module()
class AblationMeanTimeSampler(DualPlaneTemporalSampler):
    """Aggregate the sampled frames by a masked mean instead of TConv."""

    def _fuse_time(self, sampled_image, valid_mask):
        projected = self.image_adapter(sampled_image)
        weights = valid_mask.unsqueeze(-1).to(projected.dtype)
        total = (projected * weights).sum(dim=-2)
        count = weights.sum(dim=-2).clamp_min(1.0)
        return total / count
