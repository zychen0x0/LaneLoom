"""Extract road shapes by following spatial evidence before cubic fitting."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from ..backbones.road_evidence import ROAD_EVIDENCE_CHANNELS
from .lidar_shape_token_prior import LiDARShapeTokenPrior


def continuous_path_probabilities(scores, cell_width, max_step_m=4., motion_scale_m=.75):
    """Forward/backward marginals of a banded, continuous lateral path.

    Scores have shape [B,K,Y,X]. Adjacent rows exchange evidence only within
    max_step_m; missing rows may have neutral scores and receive context from
    both directions. This operates on model emissions, never annotations.
    """
    radius = min(scores.size(-1) - 1, max(1, math.ceil(max_step_m / cell_width)))
    offsets = torch.arange(-radius, radius + 1, device=scores.device, dtype=scores.dtype)
    cost = -(offsets * cell_width / motion_scale_m).square() / 2

    def message(value):
        neighbors = F.pad(value, (radius, radius), value=-torch.inf).unfold(-1, 2 * radius + 1, 1)
        return torch.logsumexp(neighbors + cost, -1)

    forward = [scores[..., 0, :]]
    for row in range(1, scores.size(-2)):
        value = scores[..., row, :] + message(forward[-1])
        forward.append(value - value.logsumexp(-1, keepdim=True))
    backward = [torch.zeros_like(forward[0])]
    for row in range(scores.size(-2) - 2, -1, -1):
        value = message(scores[..., row + 1, :] + backward[-1])
        backward.append(value - value.logsumexp(-1, keepdim=True))
    return (torch.stack(forward, -2) + torch.stack(backward[::-1], -2)).softmax(-1)


def fit_observed_cubic(values, weights, y, length, curvature_prior=None, curvature_precision=None):
    """Fit observed rows, optionally supported by predicted control differences."""
    t = y.double() / length
    basis = torch.stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3), -1)
    weighted_basis = weights.double()[..., None] * basis
    normal = torch.einsum('ri,...rj->...ij', basis, weighted_basis)
    rhs = torch.einsum('...ri,...r->...i', weighted_basis, values.double())
    if curvature_prior is not None:
        if (curvature_prior.shape != (*values.shape[:-1], 2) or curvature_precision is None
                or curvature_precision.shape != values.shape[:-1]):
            raise ValueError('Curvature prior/precision must provide two differences and one precision per curve')
        second = basis.new_tensor([[1., -2., 1., 0.], [0., 1., -2., 1.]])
        precision = curvature_precision.double()
        normal = normal + precision[..., None, None] * (second.T @ second)
        rhs = rhs + precision[..., None] * torch.einsum('ci,...c->...i', second, curvature_prior.double())
    elif curvature_precision is not None:
        raise ValueError('Curvature precision requires a predicted prior')
    normal = normal + torch.eye(4, device=values.device, dtype=torch.float64) * 1e-6
    return torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1).to(values.dtype)


def _block(in_channels, out_channels, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                  padding=dilation, dilation=dilation, bias=False),
        nn.GroupNorm(8, out_channels), nn.GELU())


@MODELS.register_module()
class ContinuousBEVShapePrior(LiDARShapeTokenPrior):
    """Five road trajectories -> five origin-relative global cubic templates.

    The 101-seed expander and downstream decoder keep their existing geometry.
    Physical coordinates enter curve fitting directly instead of being inferred
    from one pooled latent vector. Training-only auxiliary losses supervise
    visible heading/curvature, without pretending lane labels are curb labels.
    """

    def __init__(self, pc_range, longitudinal_rows=50, max_step_m=4.,
                 motion_scale_m=.75, direction_loss_weight=.1,
                 curvature_loss_weight=.05, **kwargs):
        super().__init__(**kwargs)
        channels = self.shape_tokens.embedding_dim
        if channels % 8 or longitudinal_rows < 4:
            raise ValueError('Use hidden_channels divisible by eight and at least four rows')
        if max_step_m <= 0 or motion_scale_m <= 0:
            raise ValueError('Path motion distances must be positive')
        self.pc_range = tuple(float(v) for v in pc_range)
        if self.pc_range[1] != 0 or self.pc_range[4] != float(self.anchor_y_steps[-1]):
            raise ValueError('Road rows must span the same 0..length interval as the templates')
        self.longitudinal_rows = int(longitudinal_rows)
        self.max_step_m = float(max_step_m)
        self.motion_scale_m = float(motion_scale_m)
        self.direction_loss_weight = float(direction_loss_weight)
        self.curvature_loss_weight = float(curvature_loss_weight)
        in_channels = self.feature_encoder[0].in_channels
        self.feature_encoder = nn.Sequential(
            _block(in_channels + ROAD_EVIDENCE_CHANNELS, channels),
            _block(channels, channels))
        del self.template_head
        self.coarse_encoder = nn.Sequential(
            _block(channels, channels, stride=2), _block(channels, channels, dilation=2))
        self.context_encoder = nn.Sequential(
            _block(channels, channels, stride=2), _block(channels, channels, dilation=4))
        self.fusion = _block(channels * 3, channels)
        self.position_encoder = nn.Conv2d(2, channels, 1)
        self.height_head = nn.Conv2d(channels, self.max_templates, 1)
        nn.init.normal_(self.height_head.weight, std=.001)
        nn.init.zeros_(self.height_head.bias)

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        del img_feats, img_metas
        if not isinstance(pts_feats, dict) or 'road_evidence' not in pts_feats:
            raise ValueError('ContinuousBEVShapePrior requires observed road_evidence')
        feature = self._feature(pts_feats)
        road = pts_feats['road_evidence'].to(feature)
        encoded = self.feature_encoder(torch.cat((feature, road), 1))
        coarse = self.coarse_encoder(encoded)
        context = self.context_encoder(coarse)
        encoded = self.fusion(torch.cat((
            encoded, F.interpolate(coarse, encoded.shape[-2:], mode='bilinear', align_corners=False),
            F.interpolate(context, encoded.shape[-2:], mode='bilinear', align_corners=False)), 1))
        rows, width = self.longitudinal_rows, feature.size(-1)
        memory = F.adaptive_avg_pool2d(encoded, (rows, width))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, rows, device=feature.device, dtype=feature.dtype),
            torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype), indexing='ij')
        memory = memory + self.position_encoder(torch.stack((xx, yy))[None])
        scores = torch.einsum('bcry,kc->bkry', memory, self.shape_tokens.weight) / math.sqrt(memory.size(1))
        pooled_road = F.adaptive_avg_pool2d(road, (rows, width))
        observed = pooled_road[:, 0] > 0
        scores = scores + torch.log(.02 + observed[:, None].to(scores))
        # Completely unobserved rows contribute no artificial image/height evidence.
        row_observed = observed.any(-1)
        scores = torch.where(row_observed[:, None, :, None], scores, 0)
        cell_width = (self.pc_range[3] - self.pc_range[0]) / width
        probabilities = continuous_path_probabilities(
            scores, cell_width, self.max_step_m, self.motion_scale_m)
        supported = probabilities * observed[:, None]
        support = supported.sum(-1)
        readout = torch.where(support[..., None] > 1e-6,
                              supported / support[..., None].clamp_min(1e-6), probabilities)
        x_coordinates = (torch.arange(width, device=feature.device, dtype=feature.dtype) + .5) * cell_width + self.pc_range[0]
        x_track = (readout * x_coordinates).sum(-1)
        minimum_z = pooled_road[:, 2] / pooled_road[:, 0].clamp_min(1e-6)
        z_field = minimum_z[:, None] + .5 * self.height_head(memory).tanh()
        z_track = (readout * z_field).sum(-1)
        length = self.pc_range[4]
        y = (torch.arange(rows, device=feature.device, dtype=feature.dtype) + .5) * length / rows
        # Confidence cannot be reduced through gradients to evade the fit.
        weights = support.detach() * row_observed[:, None]
        x = fit_observed_cubic(x_track, weights, y, length)
        z = fit_observed_cubic(z_track, weights, y, length)
        x_origin, z_origin = x[..., 0], z[..., 0]
        x, z = x - x[..., :1], z - z[..., :1]
        valid = row_observed.any(-1)
        x = x * valid[:, None, None]
        z = z * valid[:, None, None]
        slopes_x, slopes_z = torch.diff(x) * 3 / length, torch.diff(z) * 3 / length
        points = self._sample_curves(x, z)
        indices, mask = self._unique_template_indices(points)
        tokens = torch.einsum('bkry,bcry->bkrc', readout, memory).mean(-2)
        return dict(
            shape_x_controls=x, shape_z_controls=z,
            shape_lateral_slopes=slopes_x, shape_vertical_slopes=slopes_z,
            shape_control_yaw_degrees=torch.rad2deg(torch.atan(slopes_x)),
            shape_control_pitch_degrees=torch.rad2deg(torch.atan(slopes_z)),
            shape_query_features=tokens * valid[:, None, None],
            shape_templates=torch.stack((x, self.control_y.to(x).view(1, 1, 4).expand_as(x), z), -1),
            shape_points=points, unique_template_indices=indices, unique_template_mask=mask,
            from_lidar=valid, road_track_x=x_track, road_track_z=z_track,
            road_track_x_origin=x_origin, road_track_z_origin=z_origin,
            road_track_y=y, road_track_observed=row_observed,
            road_track_support=support, road_track_probabilities=probabilities)

    def loss(self, prior, lanes_3d, visibility, **kwargs):
        losses = super().loss(prior, lanes_3d, visibility, **kwargs)
        if (not self.supervise_shapes or
                (self.direction_loss_weight == 0 and self.curvature_loss_weight == 0)):
            return losses
        y = prior['road_track_y']
        target_y = self.anchor_y_steps.to(y)
        left = torch.searchsorted(y, target_y).sub(1).clamp(0, len(y) - 2)
        fraction = (target_y - y[left]) / (y[left + 1] - y[left])
        track = torch.stack((prior['road_track_x'], prior['road_track_z']), -1)
        prediction = track[..., left, :] * (1 - fraction[:, None]) + track[..., left + 1, :] * fraction[:, None]
        observed = prior['road_track_observed'][:, left] & prior['road_track_observed'][:, left + 1]
        valid = ((visibility > .5) & torch.isfinite(lanes_3d).all(-1)
                 & observed[:, None] & prior['from_lidar'][:, None, None])
        target = torch.where(valid[..., None], lanes_3d[..., (0, 2)], 0)
        dy = torch.diff(target_y)
        pred_direction = torch.diff(prediction, dim=-2) / dy[:, None]
        gt_direction = torch.diff(target, dim=-2) / dy[:, None]
        pair_valid = valid[..., 1:] & valid[..., :-1]

        def cost(pred, gt, mask, scales):
            difference = (pred[:, :, None] - gt[:, None]) / pred.new_tensor(scales)
            error = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none').mean(-1)
            return (error * mask[:, None]).sum(-1) / mask.sum(-1)[:, None].clamp_min(1)

        direction = cost(pred_direction, gt_direction, pair_valid, (.1, .03))
        middle_dy = (dy[1:] + dy[:-1]) * .5
        pred_curvature = torch.diff(pred_direction, dim=-2) / middle_dy[:, None]
        gt_curvature = torch.diff(gt_direction, dim=-2) / middle_dy[:, None]
        curvature = cost(pred_curvature, gt_curvature,
                         pair_valid[..., 1:] & pair_valid[..., :-1], (.002, .001))
        assignment = (direction * self.direction_loss_weight + curvature * self.curvature_loss_weight).detach().argmin(1)
        lane_valid = pair_valid.any(-1)
        for name, value, weight in (
                ('direction', direction, self.direction_loss_weight),
                ('curvature', curvature, self.curvature_loss_weight)):
            matched = value.gather(1, assignment[:, None]).squeeze(1)
            losses[f'loss_road_track_{name}'] = (matched * lane_valid).sum() / lane_valid.sum().clamp_min(1) * weight
        return losses
