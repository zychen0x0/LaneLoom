"""LiDAR-conditioned translation-invariant road-shape templates."""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from ..losses.hungarian_assigner import batched_hungarian_assignment


@MODELS.register_module()
class LaneFormerLiDARShapeTokenPrior(nn.Module):
    """Predict a compact dictionary of relative cubic-Bézier road shapes.

    Templates start at ``(x=0, y=0, z=0)``. Consequently they describe only
    shape; absolute lateral placement is delegated to proposal expansion.
    """

    def __init__(self, lidar_in_channels, pc_range, anchor_y_steps,
                 max_templates=10, hidden_channels=64,
                 lateral_slope_scale=0.1, vertical_slope_scale=0.03,
                 yaw_scale_degrees=5.0, pitch_scale_degrees=1.0,
                 curvature_scale=0.002, vertical_curvature_scale=0.001,
                 yaw_weight=1.0, pitch_weight=1.0,
                 curvature_weight=1.0, vertical_curvature_weight=1.0,
                 shape_softmin_temperature=0.25,
                 shape_nms_threshold=1.0,
                 aligned_p80_threshold=0.3,
                 yaw_similarity_degrees=1.0,
                 pitch_similarity_degrees=0.5,
                 min_lane_points=2,
                 min_mode_points=4,
                 short_support_length=20.0,
                 shape_cover_loss_weight=1.0,
                 shape_mode_loss_weight=1.0,
                 shape_diversity_loss_weight=0.05,
                 **kwargs):
        super().__init__()
        removed_options = sorted({
            'max_yaw_degrees',
            'max_pitch_degrees',
            'compact_temperature',
            'compact_loss_weight',
        } & kwargs.keys())
        if removed_options:
            names = ', '.join(removed_options)
            raise TypeError(
                f'{names} were removed from the Shape Token prior. Angle '
                'bounds were replaced by unbounded slope scaling, while '
                'compact loss was replaced by GT mode matching and '
                'matched-template diversity.')
        self.pc_range = tuple(float(value) for value in pc_range)
        self.max_templates = int(max_templates)
        self.lateral_slope_scale = float(lateral_slope_scale)
        self.vertical_slope_scale = float(vertical_slope_scale)
        self.shape_softmin_temperature = float(shape_softmin_temperature)
        self.shape_nms_threshold = float(shape_nms_threshold)
        self.aligned_p80_threshold = float(aligned_p80_threshold)
        self.yaw_similarity_threshold = math.radians(
            float(yaw_similarity_degrees))
        self.pitch_similarity_threshold = math.radians(
            float(pitch_similarity_degrees))
        self.min_lane_points = int(min_lane_points)
        self.min_mode_points = int(min_mode_points)
        self.short_support_length = float(short_support_length)
        self.shape_cover_loss_weight = float(shape_cover_loss_weight)
        self.shape_mode_loss_weight = float(shape_mode_loss_weight)
        self.shape_diversity_loss_weight = float(
            shape_diversity_loss_weight)
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
        if self.lateral_slope_scale <= 0 or self.vertical_slope_scale <= 0:
            raise ValueError('Slope scales must be positive.')
        if min(self.shape_scales) <= 0:
            raise ValueError('All shape feature scales must be positive.')
        if self.shape_softmin_temperature <= 0:
            raise ValueError('Shape softmin temperature must be positive.')
        if min(
                self.aligned_p80_threshold,
                self.yaw_similarity_threshold,
                self.pitch_similarity_threshold,
                self.short_support_length) <= 0:
            raise ValueError('Shape similarity thresholds must be positive.')
        if self.min_lane_points < 2 or self.min_mode_points < 2:
            raise ValueError('Shape losses require at least two points.')

        anchor_y_steps = torch.as_tensor(anchor_y_steps, dtype=torch.float32)
        if anchor_y_steps.ndim != 1 or anchor_y_steps.numel() < 2:
            raise ValueError('anchor_y_steps must contain at least two values.')
        if anchor_y_steps[0] < 0 or not torch.all(
                anchor_y_steps[1:] > anchor_y_steps[:-1]):
            raise ValueError(
                'Shape templates require non-negative increasing anchors.')
        self.register_buffer('anchor_y_steps', anchor_y_steps)
        length = anchor_y_steps[-1]
        t = anchor_y_steps / length
        self.register_buffer(
            'bezier_basis',
            torch.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                         3 * (1 - t) * t ** 2, t ** 3), dim=-1))
        self.register_buffer(
            'control_y', torch.linspace(0, float(length), 4))
        if self.max_templates <= 7:
            permutations = torch.tensor(
                list(itertools.permutations(range(self.max_templates))),
                dtype=torch.long)
        else:
            # Avoid factorial storage for configurations outside the intended
            # five-template Shape Prior; these fall back to SciPy Hungarian.
            permutations = torch.empty(0, self.max_templates, dtype=torch.long)
        self.register_buffer(
            'assignment_permutations', permutations, persistent=False)

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
        if isinstance(features, dict):
            features = features['features']
        return features[-1] if isinstance(features, (list, tuple)) else features

    @staticmethod
    def _lidar_sample_valid(pts_feats, batch, device):
        if isinstance(pts_feats, dict) and 'occupancy' in pts_feats:
            return pts_feats['occupancy'].flatten(1).any(dim=-1)
        return torch.ones(batch, dtype=torch.bool, device=device)

    def _relative_controls(self, pts_feats, img_feats=None):
        if pts_feats is None:
            if img_feats is None:
                raise ValueError(
                    'Shape prior requires LiDAR or image batch context.')
            image = self._feature(img_feats)
            batch = image.size(0)
            raw = image.new_zeros(batch, self.max_templates, 6)
            tokens = image.new_zeros(
                batch, self.max_templates, self.shape_tokens.embedding_dim)
            from_lidar = torch.zeros(
                batch, dtype=torch.bool, device=image.device)
        else:
            feature = self._feature(pts_feats)
            context = self.feature_encoder(feature)
            from_lidar = self._lidar_sample_valid(
                pts_feats, context.size(0), context.device)
            context = context * from_lidar[:, None].to(context.dtype)
            tokens = context.unsqueeze(1) + self.shape_tokens.weight.unsqueeze(0)
            raw = self.template_head(tokens)
        lateral_slopes = raw[..., :3] * self.lateral_slope_scale
        vertical_slopes = raw[..., 3:] * self.vertical_slope_scale
        segment_length = self.anchor_y_steps[-1].to(raw) / 3.0
        delta_x = segment_length * lateral_slopes
        delta_z = segment_length * vertical_slopes
        zeros = raw.new_zeros(raw.size(0), self.max_templates, 1)
        x_controls = torch.cat((zeros, torch.cumsum(delta_x, dim=-1)), dim=-1)
        z_controls = torch.cat((zeros, torch.cumsum(delta_z, dim=-1)), dim=-1)
        control_yaw_degrees = torch.rad2deg(torch.atan(lateral_slopes))
        control_pitch_degrees = torch.rad2deg(torch.atan(vertical_slopes))
        return (x_controls, z_controls, lateral_slopes, vertical_slopes,
                control_yaw_degrees, control_pitch_degrees, tokens,
                from_lidar)

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
        difference = torch.where(mask, difference, 0)
        count = mask.sum(dim=-1)
        distance = difference.abs().sum(dim=-1)
        distance = distance / count.clamp_min(1).to(distance.dtype)
        return distance, count > 0

    @staticmethod
    def _masked_p80(values, mask):
        count = mask.sum(dim=-1)
        ordered = values.masked_fill(~mask, torch.inf).sort(dim=-1).values
        index = (
            torch.ceil(count.to(values.dtype) * 0.8).to(torch.long) - 1
        ).clamp_min(0)
        p80 = ordered.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        return p80, count

    def _aligned_geometry_distance(
            self, templates, targets, point_mask):
        difference = (
            targets.unsqueeze(2)[..., (0, 2)]
            - templates.unsqueeze(1)[..., (0, 2)])
        count = point_mask.sum(dim=-1)
        difference = torch.where(point_mask.unsqueeze(-1), difference, 0)
        translation = difference.sum(dim=-2) / count.clamp_min(
            1).unsqueeze(-1).to(difference.dtype)
        residual = torch.linalg.vector_norm(
            difference - translation.unsqueeze(-2), dim=-1)
        residual = torch.where(point_mask, residual, 0)
        p80, _ = self._masked_p80(residual, point_mask)
        return p80, count

    def shape_distance(
            self, templates, targets, target_visibility,
            template_visibility=None, return_components=False):
        """Return shape-only distance after optimal x/z translation.

        The hard similarity interpretation is intentionally physical:
        aligned point-error P80 <= 0.3 m, mean yaw error <= 1 degree and mean
        pitch error <= 0.5 degree by default. Curvature terms remain in the
        differentiable training distance but do not redefine GT mode count.
        """
        if templates.ndim != 4 or targets.ndim != 4:
            raise ValueError('Shape distance expects [B,N,A,3] point tensors.')
        if template_visibility is None:
            template_visibility = torch.ones(
                templates.shape[:-1], dtype=torch.bool,
                device=templates.device)
        else:
            template_visibility = template_visibility > 0.5
        target_visibility = target_visibility > 0.5
        template_visibility &= torch.isfinite(templates).all(dim=-1)
        target_visibility &= torch.isfinite(targets).all(dim=-1)
        point_mask = (
            target_visibility.unsqueeze(2)
            & template_visibility.unsqueeze(1))
        aligned_p80, point_count = self._aligned_geometry_distance(
            templates, targets, point_mask)
        template_features = self._shape_features(
            templates, template_visibility)
        target_features = self._shape_features(targets, target_visibility)
        component_values = []
        component_available = []
        for ((template_value, _, angular),
             (target_value, target_mask, _), (_, template_mask, _),
             scale, weight) in zip(
                template_features, target_features,
                template_features,
                self.shape_scales, self.shape_weights):
            pair_mask = (
                target_mask.unsqueeze(2) & template_mask.unsqueeze(1))
            component, available = self._masked_component_distance(
                template_value, target_value, pair_mask, angular)
            component_values.append(component)
            component_available.append(available)

        yaw, pitch, curvature, vertical_curvature = component_values
        normalized = (
            aligned_p80 / self.aligned_p80_threshold,
            yaw / self.shape_scales[0],
            pitch / self.shape_scales[1],
            curvature / self.shape_scales[2],
            vertical_curvature / self.shape_scales[3],
        )
        weights = (1.0,) + self.shape_weights
        availability = (
            point_count >= self.min_lane_points,
            *component_available,
        )
        total = templates.new_zeros(aligned_p80.shape)
        weight_total = torch.zeros_like(total)
        for value, available, weight in zip(
                normalized, availability, weights):
            total = total + value.masked_fill(~available, 0) * weight
            weight_total = weight_total + available.to(total.dtype) * weight
        pair_valid = point_count >= self.min_lane_points
        total = (total / weight_total.clamp_min(1e-6)).masked_fill(
            ~pair_valid, torch.inf)
        valid = target_visibility.sum(dim=-1) >= self.min_lane_points
        if not return_components:
            return total, valid
        similarity_distance = torch.stack((
            aligned_p80 / self.aligned_p80_threshold,
            yaw / self.yaw_similarity_threshold,
            pitch / self.pitch_similarity_threshold,
        ), dim=-1).amax(dim=-1).masked_fill(~pair_valid, torch.inf)
        components = dict(
            aligned_p80=aligned_p80,
            yaw=yaw,
            pitch=pitch,
            curvature=curvature,
            vertical_curvature=vertical_curvature,
            point_count=point_count,
            pair_valid=pair_valid,
            similarity_distance=similarity_distance,
        )
        return total, valid, components

    @staticmethod
    def _shape_similar(components, threshold=1.0):
        return (
            components['pair_valid']
            & (components['similarity_distance'] <= threshold))

    @staticmethod
    def _softmin(distance, temperature):
        """Normalized log-mean-exp soft minimum with positive gradients."""
        count = distance.size(-1)
        return -temperature * (
            torch.logsumexp(-distance / temperature, dim=-1)
            - math.log(count))

    @staticmethod
    def _complete_link_medoids(pairwise, threshold, valid_mask=None):
        """GPU complete-link clustering followed by one medoid per cluster."""
        batch, count, _ = pairwise.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                batch, count, dtype=torch.bool, device=pairwise.device)
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
                assignment &= valid_mask[:, template_idx].unsqueeze(-1)
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
        _, _, components = self.shape_distance(
            shape_points, shape_points, visibility,
            template_visibility=visibility, return_components=True)
        return self._complete_link_medoids(
            components['similarity_distance'].detach(),
            self.shape_nms_threshold)

    def _gt_shape_modes(self, lanes_3d, visibility):
        """Approximate the GT epsilon-cover number with complete-link modes.

        This metric rank is deliberately not the algebraic rank of Bézier
        parameters: translated parallel lanes form one mode, while distinct
        straight/turn shapes require different medoids.
        """
        visibility = visibility > 0.5
        _, _, components = self.shape_distance(
            lanes_3d.float(), lanes_3d.float(), visibility,
            template_visibility=visibility, return_components=True)
        mode_valid = visibility.sum(dim=-1) >= self.min_mode_points
        pairwise = components['similarity_distance']
        enough_overlap = (
            components['point_count'] >= self.min_mode_points)
        pairwise = pairwise.masked_fill(
            ~(mode_valid.unsqueeze(1) & mode_valid.unsqueeze(2)
              & enough_overlap), torch.inf)
        medoids, active = self._complete_link_medoids(
            pairwise.detach(), threshold=1.0, valid_mask=mode_valid)
        return medoids, active, pairwise

    def _support_weights(self, visibility):
        visibility = visibility > 0.5
        y = self.anchor_y_steps.to(visibility.device)
        y_min = torch.where(
            visibility, y.view(1, 1, -1), torch.inf).amin(dim=-1)
        y_max = torch.where(
            visibility, y.view(1, 1, -1), -torch.inf).amax(dim=-1)
        span = (y_max - y_min).clamp_min(0)
        return (span / self.short_support_length).clamp(max=1.0)

    @staticmethod
    def _matched_values(matrix, assignments, active):
        safe = assignments.clamp_min(0)
        selected = matrix.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        matched = active & (assignments >= 0) & torch.isfinite(selected)
        return selected, matched

    def _assign_modes(self, mode_distance, active):
        """Exact GPU injection for <=7 templates, SciPy fallback otherwise."""
        if self.assignment_permutations.numel() == 0:
            return batched_hungarian_assignment(mode_distance)
        batch, mode_slots, template_count = mode_distance.shape
        assignments = torch.full(
            (batch, mode_slots), -1, dtype=torch.long,
            device=mode_distance.device)
        matched_count = active.sum(dim=-1).clamp_max(template_count)
        permutations = self.assignment_permutations
        for count in range(1, min(template_count, mode_slots) + 1):
            sample_mask = matched_count == count
            rows = mode_distance[:, :count]
            candidate = permutations[:, :count]
            candidate_cost = rows.unsqueeze(1).expand(
                -1, candidate.size(0), -1, -1).gather(
                    -1,
                    candidate.view(
                        1, candidate.size(0), count, 1).expand(
                            batch, -1, -1, -1)).squeeze(-1).sum(dim=-1)
            best = candidate[candidate_cost.argmin(dim=-1)]
            assignments[:, :count] = torch.where(
                sample_mask.unsqueeze(-1), best, assignments[:, :count])
        return assignments

    def _mode_losses(
            self, distance, components, prior_points, lanes_3d, visibility):
        medoids, active, _ = self._gt_shape_modes(lanes_3d, visibility)
        mode_distance = distance.gather(
            1, medoids.unsqueeze(-1).expand(-1, -1, distance.size(-1)))
        mode_distance = mode_distance.masked_fill(~active.unsqueeze(-1), torch.inf)
        assignments = self._assign_modes(mode_distance, active)
        selected, matched = self._matched_values(
            mode_distance, assignments, active)
        mode_loss = torch.where(matched, selected, 0).sum()
        mode_loss = mode_loss / matched.sum().clamp_min(1).to(selected.dtype)

        similarity = components['similarity_distance'].gather(
            1, medoids.unsqueeze(-1).expand(
                -1, -1, components['similarity_distance'].size(-1)))
        selected_similarity, similarity_matched = self._matched_values(
            similarity, assignments, active)
        mode_count = active.sum(dim=-1)
        mode_cover = (
            ((selected_similarity <= 1.0) & similarity_matched).sum(dim=-1)
            / mode_count.clamp_min(1).to(selected_similarity.dtype))

        template_visibility = torch.ones(
            prior_points.shape[:-1], dtype=torch.bool,
            device=prior_points.device)
        _, _, template_components = self.shape_distance(
            prior_points, prior_points, template_visibility,
            template_visibility=template_visibility, return_components=True)
        template_pairwise = template_components['similarity_distance']
        safe_assignments = assignments.clamp_min(0)
        matched_templates = (
            F.one_hot(safe_assignments, num_classes=prior_points.size(1))
            * matched.unsqueeze(-1)).any(dim=1)
        upper = torch.triu(
            torch.ones(
                prior_points.size(1), prior_points.size(1),
                dtype=torch.bool, device=prior_points.device),
            diagonal=1)
        pair_mask = (
            matched_templates.unsqueeze(1)
            & matched_templates.unsqueeze(2)
            & upper.unsqueeze(0))
        diversity_per_pair = F.relu(1.0 - template_pairwise).square()
        diversity = torch.where(
            pair_mask, diversity_per_pair, 0).sum()
        diversity = diversity / pair_mask.sum().clamp_min(1).to(
            diversity.dtype)
        min_distance = template_pairwise.masked_fill(
            ~pair_mask, torch.inf).amin(dim=(1, 2))
        min_distance_valid = pair_mask.flatten(1).any(dim=1)
        min_distance = torch.where(min_distance_valid, min_distance, 0)
        overflow = (
            mode_count > prior_points.size(1)).to(prior_points.dtype)
        return dict(
            mode_loss=mode_loss,
            diversity=diversity,
            mode_count=mode_count.to(prior_points.dtype),
            mode_cover=mode_cover,
            overflow=overflow,
            min_distance=min_distance,
            min_distance_valid=min_distance_valid,
        )

    def forward(self, pts_feats, img_feats=None, img_metas=None):
        (x_controls, z_controls, lateral_slopes, vertical_slopes,
         control_yaw_degrees, control_pitch_degrees,
         shape_query_features,
         from_lidar) = self._relative_controls(pts_feats, img_feats)
        shape_points = self._sample_curves(x_controls, z_controls)
        control_y = self.control_y.to(x_controls).view(1, 1, 4).expand_as(
            x_controls)
        shape_templates = torch.stack(
            (x_controls, control_y, z_controls), dim=-1)
        unique_indices, unique_mask = self._unique_template_indices(shape_points)
        return dict(
            shape_x_controls=x_controls,
            shape_z_controls=z_controls,
            shape_lateral_slopes=lateral_slopes,
            shape_vertical_slopes=vertical_slopes,
            shape_control_yaw_degrees=control_yaw_degrees,
            shape_control_pitch_degrees=control_pitch_degrees,
            shape_query_features=shape_query_features,
            shape_templates=shape_templates,
            shape_points=shape_points,
            unique_template_indices=unique_indices,
            unique_template_mask=unique_mask,
            from_lidar=from_lidar,
        )

    def loss(self, prior, lanes_3d, visibility):
        from_lidar = prior['from_lidar']
        if not torch.is_tensor(from_lidar):
            from_lidar = torch.full(
                (prior['shape_points'].size(0),), bool(from_lidar),
                dtype=torch.bool, device=prior['shape_points'].device)
        clean_visibility = (
            (visibility > 0.5) & torch.isfinite(lanes_3d).all(dim=-1))
        distance, valid, components = self.shape_distance(
            prior['shape_points'], lanes_3d.float(), clean_visibility,
            return_components=True)
        valid = valid & from_lidar.unsqueeze(-1)
        support_weight = self._support_weights(clean_visibility) * valid
        cover_per_lane = self._softmin(
            distance, self.shape_softmin_temperature).masked_fill(~valid, 0)
        cover = (
            (cover_per_lane * support_weight).sum()
            / support_weight.sum().clamp_min(1e-6))
        lane_covered = self._shape_similar(components).any(dim=-1)
        lane_cover_recall = (
            (lane_covered & valid).sum()
            / valid.sum().clamp_min(1).to(distance.dtype))

        mode = self._mode_losses(
            distance, components, prior['shape_points'],
            lanes_3d.float(),
            clean_visibility & from_lidar[:, None, None])
        unique_mask = prior.get('unique_template_mask')
        if unique_mask is None:
            _, unique_mask = self._unique_template_indices(
                prior['shape_points'])
        unique_count = unique_mask.sum(dim=-1).to(
            prior['shape_points'].dtype)
        sample_mask = from_lidar.to(prior['shape_points'].dtype)
        sample_count = sample_mask.sum().clamp_min(1)
        return dict(
            loss_lidar_shape_cover=cover * self.shape_cover_loss_weight,
            loss_lidar_shape_mode=(
                mode['mode_loss'] * self.shape_mode_loss_weight),
            loss_lidar_shape_diversity=(
                mode['diversity'] * self.shape_diversity_loss_weight),
            diag_unique_template_count=(
                unique_count * sample_mask).sum() / sample_count,
            diag_predicted_shape_metric_rank=(
                unique_count * sample_mask).sum() / sample_count,
            diag_gt_shape_metric_rank=(
                mode['mode_count'] * sample_mask).sum() / sample_count,
            diag_shape_lane_cover_recall=lane_cover_recall,
            diag_shape_mode_cover_recall=(
                mode['mode_cover'] * sample_mask).sum() / sample_count,
            diag_shape_mode_overflow_rate=(
                mode['overflow'] * sample_mask).sum() / sample_count,
            diag_matched_template_min_distance=(
                mode['min_distance'] * mode['min_distance_valid']).sum()
            / mode['min_distance_valid'].sum().clamp_min(1).to(
                mode['min_distance'].dtype),
        )
