"""LiDAR-only dynamic pillar rasterization and lightweight BEV encoding."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerDynamicPillarBEVEncoder(nn.Module):
    """Aggregate all points in x-y pillars while retaining height statistics."""

    statistic_channels = 10

    def __init__(self, point_cloud_range, pillar_size, out_channels=256,
                 queue_length=3, base_channels=64, **kwargs):
        super().__init__()
        self.point_cloud_range = tuple(float(value)
                                       for value in point_cloud_range)
        self.pillar_size = tuple(float(value) for value in pillar_size)
        self.out_channels = int(out_channels)
        self.queue_length = int(queue_length)
        if len(self.pillar_size) != 2 or min(self.pillar_size) <= 0:
            raise ValueError('pillar_size must contain positive x/y sizes.')
        x_span = self.point_cloud_range[3] - self.point_cloud_range[0]
        y_span = self.point_cloud_range[4] - self.point_cloud_range[1]
        self.grid_w = int(round(x_span / self.pillar_size[0]))
        self.grid_h = int(round(y_span / self.pillar_size[1]))
        if (not math.isclose(
                self.grid_w * self.pillar_size[0], x_span, abs_tol=1e-6)
                or not math.isclose(
                    self.grid_h * self.pillar_size[1], y_span, abs_tol=1e-6)):
            raise ValueError(
                'point_cloud_range must be divisible by pillar_size.')

        norm = dict(eps=1e-3, momentum=0.01)
        self.backbone = nn.Sequential(
            nn.Conv2d(
                self.statistic_channels, base_channels, 3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels, **norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels, base_channels * 2, 3,
                stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2, **norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 2, self.out_channels, 3,
                padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels, **norm),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _scatter_sum(values, indices, size):
        output = values.new_zeros(size)
        output.index_add_(0, indices, values)
        return output

    @staticmethod
    def _scatter_extreme(values, indices, size, reduce):
        fill = torch.inf if reduce == 'amin' else -torch.inf
        output = values.new_full((size,), fill)
        output.scatter_reduce_(
            0, indices, values, reduce=reduce, include_self=True)
        return torch.where(torch.isfinite(output), output,
                           torch.zeros_like(output))

    def _sample_raster(self, points):
        cells = self.grid_h * self.grid_w
        raster = points.new_zeros(self.statistic_channels, cells)
        if points.numel() == 0:
            return raster.view(
                self.statistic_channels, self.grid_h, self.grid_w)
        if points.ndim != 2 or points.size(-1) < 6:
            raise ValueError(
                'Dynamic pillar points must contain '
                'xyz/intensity/elongation/queue_index.')

        x_index = torch.floor(
            (points[:, 0] - self.point_cloud_range[0])
            / self.pillar_size[0]).long()
        y_index = torch.floor(
            (points[:, 1] - self.point_cloud_range[1])
            / self.pillar_size[1]).long()
        valid = (
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] <= self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] <= self.point_cloud_range[4])
            & (points[:, 2] >= self.point_cloud_range[2])
            & (points[:, 2] <= self.point_cloud_range[5]))
        if not valid.any():
            return raster.view(
                self.statistic_channels, self.grid_h, self.grid_w)

        points = points[valid]
        flat_index = (
            y_index[valid].clamp(0, self.grid_h - 1) * self.grid_w
            + x_index[valid].clamp(0, self.grid_w - 1))
        ones = points.new_ones(points.size(0))
        count = self._scatter_sum(ones, flat_index, cells)
        denominator = count.clamp_min(1)
        z = points[:, 2]
        intensity = torch.log1p(points[:, 3].clamp_min(0))
        elongation = points[:, 4]
        queue = points[:, 5] / max(self.queue_length - 1, 1)

        z_sum = self._scatter_sum(z, flat_index, cells)
        z_square_sum = self._scatter_sum(z.square(), flat_index, cells)
        z_mean = z_sum / denominator
        z_std = (
            z_square_sum / denominator - z_mean.square()).clamp_min(0).sqrt()
        intensity_mean = (
            self._scatter_sum(intensity, flat_index, cells) / denominator)
        elongation_mean = (
            self._scatter_sum(elongation, flat_index, cells) / denominator)
        queue_mean = (
            self._scatter_sum(queue, flat_index, cells) / denominator)

        raster = torch.stack((
            torch.log1p(count),
            self._scatter_extreme(z, flat_index, cells, 'amin'),
            self._scatter_extreme(z, flat_index, cells, 'amax'),
            z_mean,
            z_std,
            intensity_mean,
            self._scatter_extreme(
                intensity, flat_index, cells, 'amax'),
            elongation_mean,
            self._scatter_extreme(
                elongation, flat_index, cells, 'amax'),
            queue_mean,
        ))
        occupied = count > 0
        raster[:, ~occupied] = 0
        return raster.view(
            self.statistic_channels, self.grid_h, self.grid_w)

    def rasterize(self, points):
        """Return the deterministic LiDAR statistic image before the CNN."""
        if not isinstance(points, (list, tuple)):
            points = [points]
        if not points:
            raise ValueError('Dynamic pillar encoder requires a point batch.')
        return torch.stack(
            [self._sample_raster(sample) for sample in points], dim=0)

    def forward(self, points, img_feats=None, img_metas=None):
        del img_feats, img_metas
        raster = self.rasterize(points)
        occupancy = raster[:, :1] > 0
        features = self.backbone(raster)
        # Match the two stride-2 backbone stages. Max pooling marks a feature
        # cell valid when any raw pillar in its receptive field has points.
        occupancy = F.max_pool2d(
            occupancy.float(), kernel_size=3, stride=2, padding=1)
        occupancy = F.max_pool2d(
            occupancy, kernel_size=3, stride=2, padding=1) > 0
        sample_valid = occupancy.flatten(1).any(dim=-1)
        features = features * sample_valid[:, None, None, None].to(
            features.dtype)
        return dict(features=features, occupancy=occupancy)
