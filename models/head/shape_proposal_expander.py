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
                 q0_geometry_loss_weight=1., q0_loss_type='percentile',
                 q0_assignment='hungarian',
                 **kwargs):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.num_x_seeds = int(num_x_seeds)
        self.q0_distance_threshold = float(q0_distance_threshold)
        self.q0_match_ratio = float(q0_match_ratio)
        self.q0_geometry_loss_weight = float(q0_geometry_loss_weight)
        self.q0_loss_type = str(q0_loss_type)
        self.q0_assignment = str(q0_assignment)
        if self.num_x_seeds < 2:
            raise ValueError('num_x_seeds must include at least two endpoints.')
        if self.q0_distance_threshold <= 0:
            raise ValueError('q0_distance_threshold must be positive.')
        if not 0 < self.q0_match_ratio <= 1:
            raise ValueError('q0_match_ratio must be in (0, 1].')
        if self.q0_loss_type not in (
                'percentile', 'pointwise', 'separated_pointwise'):
            raise ValueError(
                'q0_loss_type must be "percentile", "pointwise" or '
                '"separated_pointwise".')
        if self.q0_assignment not in ('hungarian', 'per_gt_min'):
            raise ValueError(
                'q0_assignment must be "hungarian" or "per_gt_min".')
        if (self.q0_assignment == 'per_gt_min'
                and self.q0_loss_type != 'separated_pointwise'):
            raise ValueError(
                'per_gt_min assignment requires separated_pointwise loss.')

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
                          sample_valid, z_weights=None):
        lane_xz = lanes_3d.float()[..., (0, 2)]
        visible = (
            (visibility > .5) & torch.isfinite(lane_xz).all(dim=-1))
        lane_xz = torch.where(visible.unsqueeze(-1), lane_xz, 0)
        point_delta = (
            candidate_points[..., (0, 2)].unsqueeze(2)
            - lane_xz.unsqueeze(1))
        point_distance = torch.linalg.vector_norm(point_delta, dim=-1)
        visible_count = visible.sum(dim=-1)
        if z_weights is not None:
            if self.q0_loss_type != 'separated_pointwise':
                raise ValueError(
                    'z_weights require separated_pointwise Q0 loss.')
            z_weights = z_weights.to(
                device=candidate_points.device,
                dtype=candidate_points.dtype).clamp_min(0)
            z_weights = z_weights * visible.to(z_weights.dtype)
            z_weight_sum = z_weights.sum(dim=-1)
        pair_x_loss = pair_z_loss = None
        if self.q0_loss_type == 'separated_pointwise':
            x_violation = (
                point_delta[..., 0].abs() - self.q0_distance_threshold
            ).clamp_min(0)
            point_x_loss = F.smooth_l1_loss(
                x_violation, torch.zeros_like(x_violation),
                beta=self.q0_distance_threshold, reduction='none')
            point_z_loss = F.smooth_l1_loss(
                point_delta[..., 1],
                torch.zeros_like(point_delta[..., 1]),
                beta=self.q0_distance_threshold, reduction='none')
            visible_mask = visible.unsqueeze(1)
            denominator = visible_count.unsqueeze(1).clamp_min(1)
            pair_x_loss = (
                (point_x_loss * visible_mask).sum(dim=-1) / denominator)
            if z_weights is None:
                pair_z_loss = (
                    (point_z_loss * visible_mask).sum(dim=-1)
                    / denominator)
            else:
                pair_z_loss = (
                    (point_z_loss * z_weights.unsqueeze(1)).sum(dim=-1)
                    / z_weight_sum.unsqueeze(1).clamp_min(1e-6))
            pair_loss = pair_x_loss + pair_z_loss
        elif self.q0_loss_type == 'pointwise':
            violation = torch.where(
                visible.unsqueeze(1),
                (point_distance - self.q0_distance_threshold).clamp_min(0),
                0)
            point_loss = F.smooth_l1_loss(
                violation, torch.zeros_like(violation),
                beta=self.q0_distance_threshold, reduction='none')
            pair_loss = (
                (point_loss * visible.unsqueeze(1)).sum(dim=-1)
                / visible_count.unsqueeze(1).clamp_min(1))
        else:
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

        if self.q0_assignment == 'per_gt_min':
            best_query = pair_loss.argmin(dim=1)
            gather_index = best_query.unsqueeze(1)
            selected_x = pair_x_loss.gather(
                dim=1, index=gather_index).squeeze(1)
            selected_z = pair_z_loss.gather(
                dim=1, index=gather_index).squeeze(1)
            # Matching uses a per-GT mean so short and long lanes are
            # comparable. Training then averages all annotated points, so a
            # tiny fragment cannot constrain a 100 m template as strongly as
            # a long lane. Multiple fragments may select the same Q0.
            x_point_weight = (
                visible_count.to(candidate_points.dtype)
                * valid_gt.to(candidate_points.dtype))
            z_point_weight = (
                x_point_weight if z_weights is None else
                z_weight_sum * valid_gt.to(candidate_points.dtype))
            x_loss = ((selected_x * x_point_weight).sum()
                      / x_point_weight.sum().clamp_min(1))
            z_loss = ((selected_z * z_point_weight).sum()
                      / z_point_weight.sum().clamp_min(1))
            return x_loss + z_loss, x_loss, z_loss

        assignments = batched_hungarian_assignment(pair_loss)
        matched = assignments >= 0
        matched_gt = assignments.clamp_min(0).unsqueeze(-1)
        matched_loss = pair_loss.gather(
            dim=-1, index=matched_gt).squeeze(-1)
        matched_loss = torch.where(
            matched, matched_loss, torch.zeros_like(matched_loss))
        loss = matched_loss.sum() / matched.sum().clamp_min(1)
        zero = loss.detach() * 0
        return loss, zero, zero

    def loss(self, expanded, lanes_3d, visibility, z_weights=None):
        candidate_points = expanded['anchor_points']
        sample_valid = self._sample_valid(
            expanded, candidate_points.size(0), candidate_points.device)
        lane_xz = lanes_3d.float()[..., (0, 2)]
        visible = (
            (visibility > .5) & torch.isfinite(lane_xz).all(dim=-1))
        lane_xz = torch.where(visible.unsqueeze(-1), lane_xz, 0)
        point_distance = torch.linalg.vector_norm(
            candidate_points[..., (0, 2)].unsqueeze(2)
            - lane_xz.unsqueeze(1), dim=-1)
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
        geometry_loss, x_loss, z_loss = self._q0_geometry_loss(
            candidate_points, lanes_3d, visibility, sample_valid,
            z_weights=z_weights)
        return dict(
            loss_q0_geometry=(
                geometry_loss * self.q0_geometry_loss_weight),
            # MMEngine sums every returned key containing "loss". Keep these
            # detached monitoring components out of the optimization total.
            diag_q0_x_objective=x_loss.detach(),
            diag_q0_z_objective=z_loss.detach(),
            diag_q0_gt_recall=covered_gt.sum().to(
                candidate_points.dtype) / valid_gt.sum().clamp_min(1),
            diag_q0_best_distance=best_distance[valid_gt].mean()
            if valid_gt.any() else candidate_points.sum() * 0)
