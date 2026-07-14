"""Ground-surface and road-geometry priors for the LiDAR branch.

This is a PyTorch adaptation of the grid-based ideas used by GroundGrid.  It
does not depend on the upstream ROS nodelet, so it can run in the training
process and on either CPU or CUDA tensors.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class GroundGridRoadPriorExtractor(nn.Module):
    """Extract ground points and cubic-Bezier road hypotheses.

    Coordinates follow the OpenLane convention used by this project: ``x`` is
    lateral, ``y`` is forward, and ``z`` is vertical.  Input points must
    already be expressed in the current ego frame.
    """

    def __init__(self,
                 pc_range: Sequence[float],
                 anchor_y_steps: Sequence[float],
                 grid_resolution: float = 0.5,
                 max_road_hypotheses: int = 3,
                 min_ground_points: int = 2,
                 ground_height_tolerance: float = 0.25,
                 distance_tolerance: float = 0.002,
                 max_seed_variance: float = 0.08,
                 max_seed_height_above_row_min: float = 0.5,
                 propagation_steps: int = 12,
                 center_bin_half_width: float | None = None,
                 min_center_bins: int = 4,
                 direction_bins: int = 81,
                 direction_limit: float = 0.7,
                 direction_corridor_width: float = 3.0,
                 direction_min_radius: float = 8.0,
                 direction_score_ratio: float = 0.15,
                 direction_min_support: int = 20,
                 **kwargs):
        super().__init__()
        if len(pc_range) != 6:
            raise ValueError('pc_range must contain six values.')
        if grid_resolution <= 0:
            raise ValueError('grid_resolution must be positive.')
        if max_road_hypotheses <= 0:
            raise ValueError('max_road_hypotheses must be positive.')
        self.pc_range = tuple(float(value) for value in pc_range)
        self.grid_resolution = float(grid_resolution)
        self.max_road_hypotheses = int(max_road_hypotheses)
        self.min_ground_points = int(min_ground_points)
        self.ground_height_tolerance = float(ground_height_tolerance)
        self.distance_tolerance = float(distance_tolerance)
        self.max_seed_variance = float(max_seed_variance)
        self.max_seed_height_above_row_min = float(
            max_seed_height_above_row_min)
        self.propagation_steps = int(propagation_steps)
        self.center_bin_half_width = (
            float(center_bin_half_width)
            if center_bin_half_width is not None else self.grid_resolution)
        self.min_center_bins = int(min_center_bins)
        self.direction_bins = int(direction_bins)
        self.direction_limit = float(direction_limit)
        self.direction_corridor_width = float(direction_corridor_width)
        self.direction_min_radius = float(direction_min_radius)
        self.direction_score_ratio = float(direction_score_ratio)
        self.direction_min_support = int(direction_min_support)
        anchors = torch.as_tensor(anchor_y_steps, dtype=torch.float32)
        if anchors.ndim != 1 or anchors.numel() < 4:
            raise ValueError('anchor_y_steps must contain at least four values.')
        if not torch.all(anchors[1:] > anchors[:-1]):
            raise ValueError('anchor_y_steps must be strictly increasing.')
        self.register_buffer('anchor_y_steps', anchors)

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        self.grid_w = int((x_max - x_min) / self.grid_resolution)
        self.grid_h = int((y_max - y_min) / self.grid_resolution)
        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError('pc_range has an empty BEV extent.')

    def forward(self, points: Sequence[torch.Tensor]) -> Dict[str, object]:
        """Process a variable-length batch of point clouds."""
        if not isinstance(points, (list, tuple)) or not points:
            raise ValueError('points must be a non-empty sequence of tensors.')

        ground_masks: List[torch.Tensor] = []
        ground_points: List[torch.Tensor] = []
        heights, confidences = [], []
        controls, samples, valid = [], [], []
        horizontal_curvatures, vertical_curvatures, directions = [], [], []

        for sample in points:
            if sample.ndim != 2 or sample.size(-1) < 3:
                raise ValueError('each point tensor must have shape [N, >=3].')
            mask, height, confidence = self._extract_ground(sample)
            sample_ground = sample[mask]
            road = self._extract_road_hypotheses(sample_ground[:, :3])
            ground_masks.append(mask)
            ground_points.append(sample_ground)
            heights.append(height)
            confidences.append(confidence)
            controls.append(road[0])
            samples.append(road[1])
            valid.append(road[2])
            horizontal_curvatures.append(road[3])
            vertical_curvatures.append(road[4])
            directions.append(road[5])

        return {
            'ground_masks': ground_masks,
            'ground_points': ground_points,
            'ground_height': torch.stack(heights),
            'ground_confidence': torch.stack(confidences),
            'road_controls': torch.stack(controls),
            'road_samples': torch.stack(samples),
            'road_valid': torch.stack(valid),
            'horizontal_curvature': torch.stack(horizontal_curvatures),
            'vertical_curvature': torch.stack(vertical_curvatures),
            'road_directions': torch.stack(directions),
            'anchor_y_steps': self.anchor_y_steps,
        }

    def _extract_ground(
            self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor]:
        xyz = points[:, :3]
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        in_range = ((xyz[:, 0] >= x_min) & (xyz[:, 0] < x_max) &
                    (xyz[:, 1] >= y_min) & (xyz[:, 1] < y_max) &
                    (xyz[:, 2] >= z_min) & (xyz[:, 2] < z_max))
        ix = torch.floor(
            (xyz[:, 0] - x_min) / self.grid_resolution).long()
        iy = torch.floor(
            (xyz[:, 1] - y_min) / self.grid_resolution).long()
        safe_ix = ix.clamp(0, self.grid_w - 1)
        safe_iy = iy.clamp(0, self.grid_h - 1)
        flat_index = safe_iy * self.grid_w + safe_ix
        cell_count = self.grid_h * self.grid_w

        counts = xyz.new_zeros(cell_count)
        counts.scatter_add_(0, flat_index[in_range],
                            torch.ones_like(xyz[in_range, 2]))
        height_sum = xyz.new_zeros(cell_count)
        squared_height_sum = xyz.new_zeros(cell_count)
        height_sum.scatter_add_(0, flat_index[in_range], xyz[in_range, 2])
        squared_height_sum.scatter_add_(
            0, flat_index[in_range], xyz[in_range, 2].square())
        mean_height = height_sum / counts.clamp_min(1)
        variance = (squared_height_sum / counts.clamp_min(1)
                    - mean_height.square()).clamp_min(0)
        low_variance = ((counts >= self.min_ground_points) &
                        (variance <= self.max_seed_variance))
        candidate_height = mean_height.masked_fill(
            ~low_variance, float('inf')).reshape(self.grid_h, self.grid_w)
        row_minimum = candidate_height.amin(dim=1, keepdim=True)
        observed = (low_variance.reshape(self.grid_h, self.grid_w)
                    & (candidate_height <= row_minimum
                       + self.max_seed_height_above_row_min)).reshape(-1)

        height = xyz.new_zeros((1, 1, self.grid_h, self.grid_w))
        known = xyz.new_zeros((1, 1, self.grid_h, self.grid_w))
        height.view(-1)[observed] = mean_height[observed]
        known.view(-1)[observed] = 1
        kernel = xyz.new_ones((1, 1, 3, 3))
        for _ in range(self.propagation_steps):
            neighbours = F.conv2d(known, kernel, padding=1)
            neighbour_heights = F.conv2d(height * known, kernel, padding=1)
            fill = (known == 0) & (neighbours > 0)
            estimate = neighbour_heights / neighbours.clamp_min(1)
            height = torch.where(fill, estimate, height)
            known = torch.where(fill, torch.ones_like(known), known)

        point_height = height.view(-1)[flat_index]
        point_known = known.view(-1)[flat_index] > 0
        tolerance = (self.ground_height_tolerance +
                     self.distance_tolerance * xyz[:, 1].clamp_min(0))
        ground_mask = (in_range & point_known &
                       ((xyz[:, 2] - point_height).abs() <= tolerance))
        confidence = (counts.reshape(1, self.grid_h, self.grid_w) /
                      max(self.min_ground_points, 1)).clamp(max=1)
        return ground_mask, height.squeeze(0), confidence

    def _extract_road_hypotheses(
            self, ground_xyz: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        anchors = self.anchor_y_steps.to(ground_xyz)
        controls = ground_xyz.new_zeros(
            (self.max_road_hypotheses, 4, 2))
        samples = ground_xyz.new_zeros(
            (self.max_road_hypotheses, anchors.numel(), 3))
        valid = torch.zeros(
            self.max_road_hypotheses, dtype=torch.bool,
            device=ground_xyz.device)
        horizontal = ground_xyz.new_zeros(
            (self.max_road_hypotheses, anchors.numel()))
        vertical = ground_xyz.new_zeros(
            (self.max_road_hypotheses, anchors.numel()))
        directions = ground_xyz.new_zeros(self.max_road_hypotheses)
        if ground_xyz.numel() == 0:
            return controls, samples, valid, horizontal, vertical, directions

        direction_candidates, root_xy = self._direction_candidates(ground_xyz)
        direction_spread = (
            torch.stack(direction_candidates).max()
            - torch.stack(direction_candidates).min()
            if direction_candidates else ground_xyz.new_tensor(0.))
        if len(direction_candidates) >= 2 and direction_spread >= 0.20:
            for output_index, direction in enumerate(
                    direction_candidates[:self.max_road_hypotheses]):
                relative = ground_xyz[:, :2] - root_xy
                residual = relative[:, 0] - direction.tan() * relative[:, 1]
                corridor = residual.abs() <= self.direction_corridor_width
                directional_centers, directional_valid = (
                    self._centerline_observations(ground_xyz[corridor], anchors))
                if directional_valid.sum() < self.min_center_bins:
                    continue
                directional_fit = self._fit_bezier(
                    directional_centers[directional_valid],
                    anchors[directional_valid], anchors)
                if directional_fit is None:
                    continue
                (controls[output_index], samples[output_index],
                 horizontal[output_index],
                 vertical[output_index]) = directional_fit
                fitted_direction = torch.atan2(
                    samples[output_index, -1, 0],
                    samples[output_index, -1, 1].clamp_min(1e-6))
                if (fitted_direction - direction).abs() > 0.20:
                    continue
                valid[output_index] = True
                directions[output_index] = fitted_direction
            if valid.sum() >= 2:
                return (controls, samples, valid, horizontal, vertical,
                        directions)
            controls.zero_()
            samples.zero_()
            valid.zero_()
            horizontal.zero_()
            vertical.zero_()
            directions.zero_()

        # A global centerline is meaningful only for a unimodal road surface;
        # at a fork it would create a phantom road between real branches.
        center_xyz, center_valid = self._centerline_observations(
            ground_xyz, anchors)
        if center_valid.sum() < self.min_center_bins:
            return controls, samples, valid, horizontal, vertical, directions
        fitted = self._fit_bezier(center_xyz[center_valid],
                                  anchors[center_valid], anchors)
        if fitted is None:
            return controls, samples, valid, horizontal, vertical, directions
        controls[0], samples[0], horizontal[0], vertical[0] = fitted
        valid[0] = True
        directions[0] = torch.atan2(samples[0, -1, 0],
                                    samples[0, -1, 1].clamp_min(1e-6))
        return controls, samples, valid, horizontal, vertical, directions

    def _direction_candidates(
            self, ground_xyz: torch.Tensor) -> Tuple[List[torch.Tensor],
                                                     torch.Tensor]:
        """Find distance-weighted direction modes around a near-ego root.

        This mirrors the road-arm decomposition in ``road_genetare_v3.py``:
        estimate a stable root near the vehicle, weight distant support more
        heavily, smooth an angular density histogram, and retain separated
        local maxima.
        """
        near = ((ground_xyz[:, 0].abs() <= 6.0) &
                (ground_xyz[:, 1] >= 0.0) & (ground_xyz[:, 1] <= 12.0))
        if near.sum() >= 10:
            root_xy = torch.stack((
                ground_xyz[near, 0].median(),
                torch.quantile(ground_xyz[near, 1], 0.2)))
        else:
            root_xy = ground_xyz.new_zeros(2)
        relative = ground_xyz[:, :2] - root_xy
        radius = torch.linalg.vector_norm(relative, dim=-1)
        far = radius >= self.direction_min_radius
        if far.sum() < self.direction_min_support:
            return [], root_xy
        angle = torch.atan2(relative[far, 0], relative[far, 1].clamp_min(1e-6))
        weights = radius[far].clamp_max(torch.quantile(radius[far], 0.95))
        in_angle_range = ((angle >= -self.direction_limit) &
                          (angle <= self.direction_limit))
        weights = weights[in_angle_range]
        angle = angle[(angle >= -self.direction_limit) &
                      (angle <= self.direction_limit)]
        if angle.numel() == 0:
            return [], root_xy
        scaled = ((angle + self.direction_limit) /
                  (2 * self.direction_limit) * self.direction_bins)
        index = scaled.long().clamp(0, self.direction_bins - 1)
        histogram = ground_xyz.new_zeros(self.direction_bins)
        histogram.scatter_add_(0, index, weights)
        histogram = F.conv1d(
            histogram.view(1, 1, -1),
            ground_xyz.new_tensor([1., 2., 3., 2., 1.]).view(1, 1, -1),
            padding=2).view(-1)
        centers = torch.linspace(
            -self.direction_limit, self.direction_limit,
            self.direction_bins, device=ground_xyz.device,
            dtype=ground_xyz.dtype)
        order = histogram.argsort(descending=True)
        candidates: List[torch.Tensor] = []
        minimum_separation = max(0.10, 4 * self.direction_limit /
                                 self.direction_bins)
        for candidate_index in order:
            if histogram[candidate_index] <= 0:
                break
            if histogram[candidate_index] < histogram.max() * self.direction_score_ratio:
                break
            previous = histogram[(candidate_index - 1) % self.direction_bins]
            following = histogram[(candidate_index + 1) % self.direction_bins]
            if histogram[candidate_index] < previous or histogram[candidate_index] < following:
                continue
            candidate = centers[candidate_index]
            if candidate.abs() < 0.04:
                continue
            if any((candidate - chosen).abs() < minimum_separation
                   for chosen in candidates):
                continue
            candidates.append(candidate)
            if len(candidates) >= self.max_road_hypotheses:
                break
        return candidates, root_xy

    def _centerline_observations(
            self, ground_xyz: torch.Tensor,
            anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        centers = ground_xyz.new_zeros((anchors.numel(), 3))
        valid = torch.zeros(
            anchors.numel(), dtype=torch.bool, device=ground_xyz.device)
        for index, y_anchor in enumerate(anchors):
            selected = ((ground_xyz[:, 1] - y_anchor).abs() <=
                        self.center_bin_half_width)
            if selected.any():
                centers[index] = ground_xyz[selected].mean(dim=0)
                centers[index, 1] = y_anchor
                valid[index] = True
        return centers, valid

    def _fit_bezier(self, observed_xyz: torch.Tensor, observed_y: torch.Tensor,
                    sample_y: torch.Tensor) -> Tuple[torch.Tensor, ...] | None:
        y_start, y_end = sample_y[0], sample_y[-1]
        span = y_end - y_start
        if span <= 0:
            return None
        sample_t = ((sample_y - y_start) / span).clamp(0, 1)
        sample_basis = GroundGridRoadPriorExtractor._bezier_basis(sample_t)
        targets = observed_xyz[:, [0, 2]]
        # First complete the anchor profile with bounded linear extrapolation.
        # Directly fitting a 0..100 m cubic to observations ending at 30..50 m
        # is ill-conditioned and can create kilometre-scale control points.
        upper = torch.searchsorted(observed_y, sample_y).clamp(
            1, observed_y.numel() - 1)
        lower = upper - 1
        fraction = ((sample_y - observed_y[lower]) /
                    (observed_y[upper] - observed_y[lower]).clamp_min(1e-6))
        completed_targets = (
            targets[lower] * (1 - fraction.unsqueeze(-1)) +
            targets[upper] * fraction.unsqueeze(-1))
        window = min(4, observed_y.numel())
        left_slope = ((targets[window - 1] - targets[0]) /
                      (observed_y[window - 1] - observed_y[0]).clamp_min(1e-6))
        right_slope = ((targets[-1] - targets[-window]) /
                       (observed_y[-1] - observed_y[-window]).clamp_min(1e-6))
        slope_limit = targets.new_tensor([0.35, 0.10])
        left_slope = torch.maximum(
            torch.minimum(left_slope, slope_limit), -slope_limit)
        right_slope = torch.maximum(
            torch.minimum(right_slope, slope_limit), -slope_limit)
        before = sample_y < observed_y[0]
        after = sample_y > observed_y[-1]
        completed_targets[before] = (
            targets[0] + (sample_y[before] - observed_y[0]).unsqueeze(-1)
            * left_slope)
        completed_targets[after] = (
            targets[-1] + (sample_y[after] - observed_y[-1]).unsqueeze(-1)
            * right_slope)
        control = torch.linalg.lstsq(
            sample_basis, completed_targets).solution
        control[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
        control[:, 1].clamp_(self.pc_range[2], self.pc_range[5])
        if not torch.isfinite(control).all():
            return None
        xz = sample_basis @ control
        sampled = torch.stack((xz[:, 0], sample_y, xz[:, 1]), dim=-1)

        first, second = GroundGridRoadPriorExtractor._bezier_derivatives(
            control, sample_t)
        first = first / span
        second = second / span.square()
        horizontal = second[:, 0] / (1 + first[:, 0].square()).pow(1.5)
        vertical = second[:, 1] / (1 + first[:, 1].square()).pow(1.5)
        return control, sampled, horizontal, vertical

    @staticmethod
    def _bezier_basis(t: torch.Tensor) -> torch.Tensor:
        return torch.stack(((1 - t).pow(3), 3 * (1 - t).square() * t,
                            3 * (1 - t) * t.square(), t.pow(3)), dim=-1)

    @staticmethod
    def _bezier_derivatives(control: torch.Tensor,
                            t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        first = 3 * (
            (1 - t).square().unsqueeze(-1) * (control[1] - control[0]) +
            (2 * (1 - t) * t).unsqueeze(-1) * (control[2] - control[1]) +
            t.square().unsqueeze(-1) * (control[3] - control[2]))
        second = 6 * (
            (1 - t).unsqueeze(-1) * (control[2] - 2 * control[1] + control[0]) +
            t.unsqueeze(-1) * (control[3] - 2 * control[2] + control[1]))
        return first, second
