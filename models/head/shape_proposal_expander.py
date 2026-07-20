"""Dense translation of LiDAR road-shape templates into lane proposals."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

from ..losses.hungarian_assigner import batched_hungarian_assignment


@MODELS.register_module()
class ShapeProposalExpander(nn.Module):
    """Translate every relative template over inclusive lateral seeds."""

    def __init__(self, pc_range, anchor_y_steps, num_x_seeds,
                 template_feature_channels, embed_dims,
                 q0_distance_threshold=.3, q0_match_ratio=.8,
                 q0_geometry_loss_weight=1., **kwargs):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.num_x_seeds = int(num_x_seeds)
        self.q0_distance_threshold = float(q0_distance_threshold)
        self.q0_match_ratio = float(q0_match_ratio)
        self.q0_geometry_loss_weight = float(q0_geometry_loss_weight)
        if self.num_x_seeds < 2:
            raise ValueError('num_x_seeds must include at least two endpoints.')
        if self.q0_distance_threshold <= 0:
            raise ValueError('q0_distance_threshold must be positive.')
        if not 0 < self.q0_match_ratio <= 1:
            raise ValueError('q0_match_ratio must be in (0, 1].')

        anchor_y_steps = torch.as_tensor(anchor_y_steps, dtype=torch.float32)
        if (anchor_y_steps.ndim != 1 or anchor_y_steps.numel() < 2
                or not torch.all(
                    anchor_y_steps[1:] > anchor_y_steps[:-1])):
            raise ValueError(
                'anchor_y_steps must be a strictly increasing sequence.')
        self.register_buffer('anchor_y_steps', anchor_y_steps)
        length = anchor_y_steps[-1]
        t = anchor_y_steps / length
        self.register_buffer(
            'bezier_basis',
            torch.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                         3 * (1 - t) * t ** 2, t ** 3), dim=-1))
        self.register_buffer(
            'x_seeds',
            torch.linspace(
                self.pc_range[0], self.pc_range[3], self.num_x_seeds))
        self.template_adapter = nn.Linear(
            int(template_feature_channels), int(embed_dims))
        self.seed_encoder = nn.Sequential(
            nn.Linear(1, int(embed_dims)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dims), int(embed_dims)),
        )

    def _sample_curves(self, x_controls, z_controls):
        basis = self.bezier_basis.to(x_controls)
        x = torch.einsum('ac,bnc->bna', basis, x_controls)
        z = torch.einsum('ac,bnc->bna', basis, z_controls)
        y = self.anchor_y_steps.to(x).view(1, 1, -1).expand_as(x)
        return torch.stack((x, y, z), dim=-1)

    def forward(self, prior, image_features=None, img_metas=None):
        del image_features, img_metas
        shape_x = prior['shape_x_controls']
        shape_z = prior['shape_z_controls']
        shape_features = prior['shape_query_features']
        batch, templates, controls = shape_x.shape
        seeds = self.x_seeds.to(shape_x)
        curve_x = (
            shape_x.unsqueeze(2) + seeds.view(1, 1, -1, 1)).reshape(
                batch, templates * self.num_x_seeds, controls)
        curve_z = shape_z.unsqueeze(2).expand(
            -1, -1, self.num_x_seeds, -1).reshape(
                batch, templates * self.num_x_seeds, controls)

        template_features = self.template_adapter(shape_features)
        x_span = max(self.pc_range[3] - self.pc_range[0], 1e-6)
        normalized_seeds = (
            (seeds - self.pc_range[0]) / x_span * 2 - 1)
        seed_features = self.seed_encoder(
            normalized_seeds.view(-1, 1))
        query_features = (
            template_features.unsqueeze(2)
            + seed_features.view(1, 1, self.num_x_seeds, -1)).reshape(
                batch, templates * self.num_x_seeds, -1)
        template_indices = torch.arange(
            templates, device=shape_x.device).view(1, -1, 1).expand(
                batch, -1, self.num_x_seeds).reshape(batch, -1)
        seed_indices = torch.arange(
            self.num_x_seeds, device=shape_x.device).view(1, 1, -1).expand(
                batch, templates, -1).reshape(batch, -1)
        return dict(
            curve_x_controls=curve_x,
            curve_z_controls=curve_z,
            anchor_points=self._sample_curves(curve_x, curve_z),
            query_features=query_features,
            selected_template_indices=template_indices,
            selected_seed_indices=seed_indices,
        )

    @staticmethod
    def _sample_valid(expanded, batch, device):
        valid = expanded.get('from_lidar')
        if valid is None:
            return torch.ones(batch, dtype=torch.bool, device=device)
        if torch.is_tensor(valid):
            return valid.to(device=device, dtype=torch.bool)
        return torch.full(
            (batch,), bool(valid), dtype=torch.bool, device=device)

    def _q0_geometry_loss(self, candidate_points, lanes_3d, visibility,
                          sample_valid):
        point_distance = torch.linalg.vector_norm(
            candidate_points[..., (0, 2)].unsqueeze(2)
            - lanes_3d.float()[..., (0, 2)].unsqueeze(1), dim=-1)
        visible = visibility > .5
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
        valid_gt = (
            (visible_count >= 2) & sample_valid.unsqueeze(-1))
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

    def loss(self, expanded, lanes_3d, visibility):
        candidate_points = expanded['anchor_points']
        sample_valid = self._sample_valid(
            expanded, candidate_points.size(0), candidate_points.device)
        visible = visibility > .5
        point_distance = torch.linalg.vector_norm(
            candidate_points[..., (0, 2)].unsqueeze(2)
            - lanes_3d.float()[..., (0, 2)].unsqueeze(1), dim=-1)
        matched_ratio = (
            ((point_distance < self.q0_distance_threshold)
             & visible.unsqueeze(1)).sum(dim=-1)
            / visible.sum(dim=-1).unsqueeze(1).clamp_min(1))
        valid_gt = (
            (visible.sum(dim=-1) >= 2) & sample_valid.unsqueeze(-1))
        covered_gt = (
            (matched_ratio >= self.q0_match_ratio).any(dim=1) & valid_gt)
        mean_distance = (
            (point_distance * visible.unsqueeze(1)).sum(dim=-1)
            / visible.sum(dim=-1).unsqueeze(1).clamp_min(1))
        best_distance = mean_distance.min(dim=1).values
        return dict(
            loss_q0_geometry=self._q0_geometry_loss(
                candidate_points, lanes_3d, visibility, sample_valid)
            * self.q0_geometry_loss_weight,
            diag_q0_gt_recall=covered_gt.sum().to(
                candidate_points.dtype) / valid_gt.sum().clamp_min(1),
            diag_q0_best_distance=best_distance[valid_gt].mean()
            if valid_gt.any() else candidate_points.sum() * 0)
