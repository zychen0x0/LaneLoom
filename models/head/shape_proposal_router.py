"""Image-evidence routing for translated LiDAR shape templates."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from ..losses.hungarian_assigner import batched_hungarian_assignment


@MODELS.register_module()
class LaneLoomShapeProposalRouter(nn.Module):
    """Expand unique shapes over x seeds and route a fixed proposal budget."""

    def __init__(self, image_in_channels, embed_dims, pc_range,
                 anchor_y_steps, num_x_seeds=50, proposal_budget=50,
                 hidden_channels=64, q0_distance_threshold=0.3,
                 q0_match_ratio=0.8, q0_geometry_loss_weight=1.0,
                 routing_loss_weight=1.0, focal_alpha=0.75,
                 focal_gamma=2.0, **kwargs):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.num_x_seeds = int(num_x_seeds)
        self.proposal_budget = int(proposal_budget)
        self.q0_distance_threshold = float(q0_distance_threshold)
        self.q0_match_ratio = float(q0_match_ratio)
        self.q0_geometry_loss_weight = float(q0_geometry_loss_weight)
        self.routing_loss_weight = float(routing_loss_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        if self.num_x_seeds <= 0 or self.proposal_budget <= 0:
            raise ValueError('Seed count and proposal budget must be positive.')
        if self.proposal_budget > self.num_x_seeds:
            raise ValueError(
                'proposal_budget cannot exceed the one-shape candidate count.')
        if self.q0_distance_threshold <= 0:
            raise ValueError('q0_distance_threshold must be positive.')
        if not 0 < self.q0_match_ratio <= 1:
            raise ValueError('q0_match_ratio must be in (0, 1].')

        anchor_y_steps = torch.as_tensor(anchor_y_steps, dtype=torch.float32)
        if anchor_y_steps.ndim != 1 or anchor_y_steps.numel() < 2:
            raise ValueError('anchor_y_steps must be one-dimensional.')
        self.register_buffer('anchor_y_steps', anchor_y_steps)
        length = anchor_y_steps[-1]
        t = anchor_y_steps / length
        self.register_buffer(
            'bezier_basis',
            torch.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                         3 * (1 - t) * t ** 2, t ** 3), dim=-1))
        x_min, x_max = self.pc_range[0], self.pc_range[3]
        seed_width = (x_max - x_min) / self.num_x_seeds
        self.register_buffer(
            'x_seeds',
            x_min + (torch.arange(self.num_x_seeds) + 0.5) * seed_width)

        self.score_head = nn.Sequential(
            nn.Linear(image_in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )
        self.query_adapter = nn.Linear(image_in_channels, embed_dims)
        nn.init.zeros_(self.score_head[-1].bias)

    @staticmethod
    def _current_image_level(image_features):
        if image_features is None:
            return None
        feature = image_features[0] if isinstance(
            image_features, (list, tuple)) else image_features
        if feature.ndim == 6:  # [B,T,N,C,H,W]
            return feature[:, -1]
        if feature.ndim == 5:  # [B,N,C,H,W]
            return feature
        if feature.ndim == 4:  # [B,C,H,W]
            return feature.unsqueeze(1)
        raise ValueError(
            'Router image feature must have shape [B,T,N,C,H,W], '
            f'[B,N,C,H,W] or [B,C,H,W], got {tuple(feature.shape)}.')

    @staticmethod
    def _current_meta(img_metas, batch_idx):
        if img_metas is None:
            return {}
        sample = (img_metas[batch_idx]
                  if isinstance(img_metas, (list, tuple)) else img_metas)
        return sample.get('current', sample) if isinstance(sample, dict) else {}

    def _sample_curves(self, x_controls, z_controls):
        basis = self.bezier_basis.to(x_controls)
        x = torch.einsum('ac,bnc->bna', basis, x_controls)
        z = torch.einsum('ac,bnc->bna', basis, z_controls)
        y = self.anchor_y_steps.to(x).view(1, 1, -1).expand_as(x)
        return torch.stack((x, y, z), dim=-1)

    def _expand_candidates(self, prior):
        shape_x = prior['shape_x_controls']
        shape_z = prior['shape_z_controls']
        batch, max_templates, controls = shape_x.shape
        unique = prior['unique_template_indices']
        unique_mask = prior['unique_template_mask']
        if unique.shape != (batch, max_templates):
            raise ValueError(
                'unique_template_indices must have shape '
                f'{(batch, max_templates)}, got {tuple(unique.shape)}.')
        selected_x = torch.gather(
            shape_x, 1, unique.unsqueeze(-1).expand(-1, -1, controls))
        selected_z = torch.gather(
            shape_z, 1, unique.unsqueeze(-1).expand(-1, -1, controls))
        seeds = self.x_seeds.to(shape_x)
        candidate_x = (
            selected_x.unsqueeze(2) + seeds.view(1, 1, -1, 1)).reshape(
                batch, max_templates * self.num_x_seeds, controls)
        candidate_z = selected_z.unsqueeze(2).expand(
            -1, -1, self.num_x_seeds, -1).reshape(
                batch, max_templates * self.num_x_seeds, controls)
        candidate_valid = unique_mask.unsqueeze(-1).expand(
            -1, -1, self.num_x_seeds).reshape(batch, -1)
        template_indices = unique.unsqueeze(-1).expand(
            -1, -1, self.num_x_seeds).reshape(batch, -1)
        seed_indices = torch.arange(
            self.num_x_seeds, device=shape_x.device).view(1, 1, -1).expand(
                batch, max_templates, -1).reshape(batch, -1)
        return (candidate_x, candidate_z, candidate_valid,
                template_indices, seed_indices)

    def _sample_current_image(self, candidate_points, candidate_valid,
                              image_features, img_metas):
        current = self._current_image_level(image_features)
        batch, candidates, anchors, _ = candidate_points.shape
        channels = self.score_head[0].in_features
        if current is None:
            return candidate_points.new_zeros(batch, candidates, channels)
        if current.size(2) != channels:
            raise ValueError(
                f'Router expected {channels} image channels, got {current.size(2)}.')

        sampled_sum = candidate_points.new_zeros(
            batch, candidates, anchors, channels)
        weight_sum = candidate_points.new_zeros(
            batch, candidates, anchors, 1)
        homogeneous = torch.cat((
            candidate_points, torch.ones_like(candidate_points[..., :1])), -1)
        for batch_idx in range(batch):
            meta = self._current_meta(img_metas, batch_idx)
            matrices = meta.get('lane2img') or meta.get('lidar2img', [])
            image_shape = meta.get('img_shape', current.shape[-2:])
            image_h, image_w = float(image_shape[0]), float(image_shape[1])
            for camera_idx in range(current.size(1)):
                matrix = (matrices[min(camera_idx, len(matrices) - 1)]
                          if matrices else torch.eye(4))
                matrix = torch.as_tensor(
                    matrix, device=candidate_points.device,
                    dtype=candidate_points.dtype)
                projected = homogeneous[batch_idx] @ matrix.transpose(0, 1)
                depth = projected[..., 2]
                grid = torch.stack((
                    projected[..., 0] / depth.clamp_min(1e-5)
                    / max(image_w - 1, 1) * 2 - 1,
                    projected[..., 1] / depth.clamp_min(1e-5)
                    / max(image_h - 1, 1) * 2 - 1,
                ), dim=-1)
                valid = ((depth > 1e-5)
                         & (grid[..., 0].abs() <= 1)
                         & (grid[..., 1].abs() <= 1)
                         & candidate_valid[batch_idx].unsqueeze(-1))
                sampled = F.grid_sample(
                    current[batch_idx, camera_idx].unsqueeze(0),
                    grid.unsqueeze(0), align_corners=True)
                sampled = sampled.squeeze(0).permute(1, 2, 0)
                weight = valid.to(sampled.dtype).unsqueeze(-1)
                sampled_sum[batch_idx] += sampled * weight
                weight_sum[batch_idx] += weight
        sampled = sampled_sum / weight_sum.clamp_min(1)
        return sampled.sum(dim=2) / weight_sum.gt(0).sum(dim=2).clamp_min(1)

    def forward(self, prior, image_features, img_metas=None):
        (candidate_x, candidate_z, candidate_valid,
         template_indices, seed_indices) = self._expand_candidates(prior)
        candidate_points = self._sample_curves(candidate_x, candidate_z)
        pooled = self._sample_current_image(
            candidate_points, candidate_valid, image_features, img_metas)
        candidate_logits = self.score_head(pooled).squeeze(-1)
        ranking_logits = candidate_logits.masked_fill(~candidate_valid, -torch.inf)
        selected_logits, selected_indices = ranking_logits.topk(
            self.proposal_budget, dim=-1)

        def gather(tensor):
            index = selected_indices
            while index.ndim < tensor.ndim:
                index = index.unsqueeze(-1)
            return torch.gather(
                tensor, 1, index.expand(-1, -1, *tensor.shape[2:]))

        selected_x = gather(candidate_x)
        selected_z = gather(candidate_z)
        selected_points = gather(candidate_points)
        selected_features = gather(self.query_adapter(pooled))
        return dict(
            curve_x_controls=selected_x,
            curve_z_controls=selected_z,
            anchor_points=selected_points,
            query_features=selected_features,
            routing_scores=selected_logits.sigmoid(),
            selected_candidate_indices=selected_indices,
            selected_template_indices=gather(template_indices),
            selected_seed_indices=gather(seed_indices),
            candidate_x_controls=candidate_x,
            candidate_z_controls=candidate_z,
            candidate_points=candidate_points,
            candidate_routing_logits=candidate_logits,
            candidate_valid_mask=candidate_valid,
        )

    def _strict_candidate_matches(self, candidate_points, candidate_valid,
                                  lanes_3d, visibility):
        """Measure Q0 candidates using stricter-than-evaluation point overlap."""
        gt_valid = (visibility > 0.5).sum(dim=-1) >= 2
        weights = (visibility > 0.5).to(candidate_points.dtype)
        candidate_xz = candidate_points[..., (0, 2)]
        gt_xz = lanes_3d.float()[..., (0, 2)]
        difference = candidate_xz.unsqueeze(2) - gt_xz.unsqueeze(1)
        point_distance = torch.linalg.vector_norm(difference, dim=-1)
        distance = (point_distance * weights.unsqueeze(1)).sum(dim=-1)
        distance = distance / weights.sum(dim=-1).unsqueeze(1).clamp_min(1)
        distance = distance.masked_fill(~gt_valid.unsqueeze(1), torch.inf)
        distance = distance.masked_fill(~candidate_valid.unsqueeze(-1), torch.inf)
        matched = (
            (point_distance < self.q0_distance_threshold)
            & (visibility > 0.5).unsqueeze(1))
        matched_ratio = (
            matched.sum(dim=-1)
            / (visibility > 0.5).sum(dim=-1).unsqueeze(1).clamp_min(1))
        strict = (
            (matched_ratio >= self.q0_match_ratio)
            & gt_valid.unsqueeze(1)
            & candidate_valid.unsqueeze(-1))
        return strict, distance, gt_valid

    def _routing_assignments(self, candidate_points, candidate_valid,
                             lanes_3d, visibility):
        """Globally maximize strict Q0 matches, then minimize distance."""
        strict, distance, gt_valid = self._strict_candidate_matches(
            candidate_points, candidate_valid, lanes_3d, visibility)
        cost = distance + (~strict).to(distance.dtype) * 1e6
        cost = cost.masked_fill(~candidate_valid.unsqueeze(-1), torch.inf)
        cost = cost.masked_fill(~gt_valid.unsqueeze(1), torch.inf)
        return batched_hungarian_assignment(cost)

    def _routing_targets(self, candidate_points, candidate_valid,
                         lanes_3d, visibility):
        assignments = self._routing_assignments(
            candidate_points, candidate_valid, lanes_3d, visibility)
        return (assignments >= 0) & candidate_valid

    def _q0_geometry_loss(self, candidate_points, candidate_valid,
                          lanes_3d, visibility):
        """Require one Q0 per GT to beat the evaluator's pointwise standard."""
        point_distance = torch.linalg.vector_norm(
            candidate_points[..., (0, 2)].unsqueeze(2)
            - lanes_3d.float()[..., (0, 2)].unsqueeze(1), dim=-1)
        visible = visibility > 0.5
        visible_count = visible.sum(dim=-1)
        required_count = torch.ceil(
            visible_count * self.q0_match_ratio).long().clamp_min(1)
        sorted_distance = point_distance.masked_fill(
            ~visible.unsqueeze(1), torch.inf).sort(dim=-1).values
        required_index = (required_count - 1).unsqueeze(1).unsqueeze(-1)
        required_index = required_index.expand(
            -1, candidate_points.size(1), -1, -1)
        percentile_distance = sorted_distance.gather(
            dim=-1, index=required_index).squeeze(-1)
        violation = (
            percentile_distance - self.q0_distance_threshold).clamp_min(0)
        pair_loss = F.smooth_l1_loss(
            violation, torch.zeros_like(violation),
            beta=self.q0_distance_threshold, reduction='none')
        pair_loss = pair_loss.masked_fill(
            ~candidate_valid.unsqueeze(-1), torch.inf)
        valid_gt = (
            (visible_count >= 2)
            & candidate_valid.any(dim=-1).unsqueeze(-1))
        pair_loss = pair_loss.masked_fill(
            ~valid_gt.unsqueeze(1), torch.inf)
        assignments = batched_hungarian_assignment(pair_loss)
        matched = assignments >= 0
        matched_gt = assignments.clamp_min(0).unsqueeze(-1)
        matched_loss = pair_loss.gather(
            dim=-1, index=matched_gt).squeeze(-1)
        matched_loss = torch.where(
            matched, matched_loss, torch.zeros_like(matched_loss))
        return matched_loss.sum() / matched.sum().clamp_min(1)

    def loss(self, routed, lanes_3d, visibility):
        candidate_valid = routed['candidate_valid_mask']
        positive = self._routing_targets(
            routed['candidate_points'], candidate_valid, lanes_3d, visibility)

        logits = routed['candidate_routing_logits']
        targets = positive.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probability = logits.sigmoid()
        pt = torch.where(targets > 0.5, probability, 1 - probability)
        alpha = torch.where(
            targets > 0.5,
            logits.new_tensor(self.focal_alpha),
            logits.new_tensor(1 - self.focal_alpha))
        focal = alpha * (1 - pt).pow(self.focal_gamma) * bce
        valid_count = candidate_valid.sum(dim=-1)
        per_sample = (
            (focal * candidate_valid.to(focal.dtype)).sum(dim=-1)
            / valid_count.clamp_min(1).to(focal.dtype))
        loss = per_sample.mean()
        q0_geometry = self._q0_geometry_loss(
            routed['anchor_points'],
            torch.ones_like(routed['routing_scores'], dtype=torch.bool),
            lanes_3d, visibility)
        return dict(
            loss_proposal_router=loss * self.routing_loss_weight,
            loss_q0_geometry=(
                q0_geometry * self.q0_geometry_loss_weight))
