"""LiDAR-conditioned translation-invariant road-shape templates."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerLiDARShapeTokenPrior(nn.Module):
    """Predict a compact dictionary of relative cubic-Bézier road shapes.

    Templates start at ``(x=0, y=0, z=0)``.  Consequently they describe only
    shape; absolute lateral placement is delegated to the proposal router.
    """

    def __init__(self, lidar_in_channels, pc_range, anchor_y_steps,
                 max_templates=10, hidden_channels=64,
                 max_yaw_degrees=30.0, max_pitch_degrees=3.0,
                 yaw_scale_degrees=5.0, pitch_scale_degrees=1.0,
                 curvature_scale=0.002, vertical_curvature_scale=0.001,
                 yaw_weight=1.0, pitch_weight=1.0,
                 curvature_weight=1.0, vertical_curvature_weight=1.0,
                 shape_softmin_temperature=0.25,
                 compact_temperature=0.25,
                 shape_nms_threshold=1.0,
                 shape_cover_loss_weight=1.0,
                 compact_loss_weight=0.05,
                 **kwargs):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.max_templates = int(max_templates)
        self.max_yaw_degrees = float(max_yaw_degrees)
        self.max_pitch_degrees = float(max_pitch_degrees)
        self.shape_softmin_temperature = float(shape_softmin_temperature)
        self.compact_temperature = float(compact_temperature)
        self.shape_nms_threshold = float(shape_nms_threshold)
        self.shape_cover_loss_weight = float(shape_cover_loss_weight)
        self.compact_loss_weight = float(compact_loss_weight)
        self.shape_scales = (
            math.radians(float(yaw_scale_degrees)),
            math.radians(float(pitch_scale_degrees)),
            float(curvature_scale),
            float(vertical_curvature_scale),
        )
        self.shape_weights = (
            float(yaw_weight), float(pitch_weight),
            float(curvature_weight), float(vertical_curvature_weight),
        )
        if self.max_templates <= 0:
            raise ValueError('max_templates must be positive.')
        if self.max_yaw_degrees <= 0 or self.max_pitch_degrees <= 0:
            raise ValueError('Maximum yaw and pitch must be positive.')
        if min(self.shape_scales) <= 0:
            raise ValueError('All shape feature scales must be positive.')
        if self.shape_softmin_temperature <= 0 or self.compact_temperature <= 0:
            raise ValueError('Shape temperatures must be positive.')

        anchor_y_steps = torch.as_tensor(anchor_y_steps, dtype=torch.float32)
        if anchor_y_steps.ndim != 1 or anchor_y_steps.numel() < 2:
            raise ValueError('anchor_y_steps must contain at least two values.')
        if anchor_y_steps[0] != 0 or not torch.all(
                anchor_y_steps[1:] > anchor_y_steps[:-1]):
            raise ValueError(
                'Shape templates require increasing anchors starting at y=0.')
        self.register_buffer('anchor_y_steps', anchor_y_steps)
        length = anchor_y_steps[-1]
        t = anchor_y_steps / length
        self.register_buffer(
            'bezier_basis',
            torch.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                         3 * (1 - t) * t ** 2, t ** 3), dim=-1))
        self.register_buffer(
            'control_y', torch.linspace(0, float(length), 4))

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(lidar_in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden_channels * 16, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.shape_tokens = nn.Embedding(self.max_templates, hidden_channels)
        self.template_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 6),
        )
        nn.init.normal_(self.shape_tokens.weight, std=0.02)
        nn.init.xavier_uniform_(self.template_head[-1].weight, gain=0.05)
        nn.init.zeros_(self.template_head[-1].bias)

    @staticmethod
    def _feature(features):
        return features[-1] if isinstance(features, (list, tuple)) else features

    def _relative_controls(self, pts_feats, img_feats=None):
        if pts_feats is None:
            if img_feats is None:
                raise ValueError(
                    'Shape prior requires LiDAR or image batch context.')
            image = self._feature(img_feats)
            batch = image.size(0)
            raw = image.new_zeros(batch, self.max_templates, 6)
            from_lidar = False
        else:
            context = self.feature_encoder(self._feature(pts_feats))
            tokens = context.unsqueeze(1) + self.shape_tokens.weight.unsqueeze(0)
            raw = self.template_head(tokens)
            from_lidar = True
        yaw = torch.tanh(raw[..., :3]) * self.max_yaw_degrees
        pitch = torch.tanh(raw[..., 3:]) * self.max_pitch_degrees
        segment_length = self.anchor_y_steps[-1].to(raw) / 3.0
        delta_x = segment_length * torch.tan(torch.deg2rad(yaw))
        delta_z = segment_length * torch.tan(torch.deg2rad(pitch))
        zeros = raw.new_zeros(raw.size(0), self.max_templates, 1)
        x_controls = torch.cat((zeros, torch.cumsum(delta_x, dim=-1)), dim=-1)
        z_controls = torch.cat((zeros, torch.cumsum(delta_z, dim=-1)), dim=-1)
        return x_controls, z_controls, yaw, pitch, from_lidar

    def _sample_curves(self, x_controls, z_controls):
        basis = self.bezier_basis.to(x_controls)
        x = torch.einsum('ac,bkc->bka', basis, x_controls)
        z = torch.einsum('ac,bkc->bka', basis, z_controls)
        y = self.anchor_y_steps.to(x).view(1, 1, -1).expand_as(x)
        return torch.stack((x, y, z), dim=-1)

    @staticmethod
    def _shape_features(points, visibility):
        delta = points[..., 1:, :] - points[..., :-1, :]
        dy = delta[..., 1].clamp_min(1e-6)
        yaw = torch.atan2(delta[..., 0], dy)
        pitch = torch.atan2(delta[..., 2], dy)
        segment_valid = visibility[..., 1:] & visibility[..., :-1]
        curvature_dy = ((dy[..., 1:] + dy[..., :-1]) * 0.5).clamp_min(1e-6)
        yaw_delta = torch.atan2(
            torch.sin(yaw[..., 1:] - yaw[..., :-1]),
            torch.cos(yaw[..., 1:] - yaw[..., :-1]))
        pitch_delta = torch.atan2(
            torch.sin(pitch[..., 1:] - pitch[..., :-1]),
            torch.cos(pitch[..., 1:] - pitch[..., :-1]))
        curvature = yaw_delta / curvature_dy
        vertical_curvature = pitch_delta / curvature_dy
        curvature_valid = segment_valid[..., 1:] & segment_valid[..., :-1]
        return (
            (yaw, segment_valid, True),
            (pitch, segment_valid, True),
            (curvature, curvature_valid, False),
            (vertical_curvature, curvature_valid, False),
        )

    @staticmethod
    def _masked_component_distance(template, target, mask, angular):
        difference = target.unsqueeze(2) - template.unsqueeze(1)
        if angular:
            difference = torch.atan2(torch.sin(difference), torch.cos(difference))
        mask = mask.unsqueeze(2)
        count = mask.sum(dim=-1)
        distance = (
            difference.abs() * mask.to(difference.dtype)).sum(dim=-1)
        distance = distance / count.clamp_min(1).to(distance.dtype)
        return distance, count > 0

    def shape_distance(self, templates, targets, target_visibility):
        """Return translation-invariant target-to-template shape distance."""
        if templates.ndim != 4 or targets.ndim != 4:
            raise ValueError('Shape distance expects [B,N,A,3] point tensors.')
        template_visibility = torch.ones(
            templates.shape[:-1], dtype=torch.bool, device=templates.device)
        target_visibility = target_visibility > 0.5
        template_features = self._shape_features(
            templates, template_visibility)
        target_features = self._shape_features(targets, target_visibility)
        total = templates.new_zeros(
            templates.size(0), targets.size(1), templates.size(1))
        weight_total = torch.zeros_like(total)
        for ((template_value, _, angular),
             (target_value, target_mask, _), scale, weight) in zip(
                template_features, target_features,
                self.shape_scales, self.shape_weights):
            component, available = self._masked_component_distance(
                template_value, target_value, target_mask, angular)
            total = total + component / scale * weight
            weight_total = weight_total + available.to(total.dtype) * weight
        valid = target_features[0][1].any(dim=-1)
        return total / weight_total.clamp_min(1e-6), valid

    @staticmethod
    def _softmin(distance, temperature):
        """Normalized log-mean-exp soft minimum with positive gradients."""
        count = distance.size(-1)
        return -temperature * (
            torch.logsumexp(-distance / temperature, dim=-1)
            - math.log(count))

    @staticmethod
    def _complete_link_medoids(pairwise, threshold):
        """GPU complete-link clustering followed by one medoid per cluster."""
        batch, count, _ = pairwise.shape
        members = torch.zeros(
            batch, count, count, dtype=torch.bool, device=pairwise.device)
        active = torch.zeros(
            batch, count, dtype=torch.bool, device=pairwise.device)
        with torch.no_grad():
            for template_idx in range(count):
                compatible = (
                    (pairwise[:, template_idx].unsqueeze(1) <= threshold)
                    | ~members).all(dim=-1) & active
                has_compatible = compatible.any(dim=-1)
                existing_cluster = compatible.to(torch.int64).argmax(dim=-1)
                new_cluster = active.sum(dim=-1).clamp_max(count - 1)
                cluster_idx = torch.where(
                    has_compatible, existing_cluster, new_cluster)
                assignment = F.one_hot(
                    cluster_idx, num_classes=count).to(torch.bool)
                template = F.one_hot(
                    torch.full(
                        (batch,), template_idx, device=pairwise.device),
                    num_classes=count).to(torch.bool)
                members |= assignment.unsqueeze(-1) & template.unsqueeze(1)
                active |= assignment

            costs = torch.einsum(
                'bij,bcj->bci', pairwise, members.to(pairwise.dtype))
            costs = costs.masked_fill(~members, torch.inf)
            medoids = costs.argmin(dim=-1)
        return medoids, active

    def _unique_template_indices(self, shape_points):
        visibility = torch.ones(
            shape_points.shape[:-1], device=shape_points.device,
            dtype=torch.bool)
        pairwise, _ = self.shape_distance(
            shape_points, shape_points, visibility)
        return self._complete_link_medoids(
            pairwise.detach(), self.shape_nms_threshold)

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        x_controls, z_controls, yaw, pitch, from_lidar = self._relative_controls(
            pts_feats, img_feats)
        shape_points = self._sample_curves(x_controls, z_controls)
        control_y = self.control_y.to(x_controls).view(1, 1, 4).expand_as(
            x_controls)
        shape_templates = torch.stack(
            (x_controls, control_y, z_controls), dim=-1)
        unique_indices, unique_mask = self._unique_template_indices(shape_points)
        return dict(
            shape_x_controls=x_controls,
            shape_z_controls=z_controls,
            shape_yaw=yaw,
            shape_pitch=pitch,
            shape_templates=shape_templates,
            shape_points=shape_points,
            unique_template_indices=unique_indices,
            unique_template_mask=unique_mask,
            from_lidar=from_lidar,
        )

    def loss(self, prior, lanes_3d, visibility):
        if not prior['from_lidar']:
            zero = prior['shape_points'].sum() * 0
            return dict(
                loss_lidar_shape_cover=zero,
                loss_lidar_shape_compact=zero)
        distance, valid = self.shape_distance(
            prior['shape_points'], lanes_3d.float(), visibility)
        if valid.any():
            cover = self._softmin(
                distance, self.shape_softmin_temperature)[valid].mean()
        else:
            cover = prior['shape_points'].sum() * 0

        template_visibility = torch.ones(
            prior['shape_points'].shape[:-1], dtype=torch.bool,
            device=prior['shape_points'].device)
        pairwise, _ = self.shape_distance(
            prior['shape_points'], prior['shape_points'], template_visibility)
        similarity = torch.exp(-pairwise / self.compact_temperature)
        effective_count = (1.0 / similarity.sum(dim=-1).clamp_min(1e-6)).sum(
            dim=-1).mean()
        return dict(
            loss_lidar_shape_cover=cover * self.shape_cover_loss_weight,
            loss_lidar_shape_compact=effective_count * self.compact_loss_weight,
        )
