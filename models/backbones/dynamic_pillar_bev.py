"""Statistics-enhanced dynamic pillars with a sparse 2D lane backbone."""

import math

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


XYZ = slice(0, 3)
X = 0
Y = 1
Z = 2
INTENSITY = 3
ELONGATION = 4
QUEUE_INDEX = 5
BASE_POINT_CHANNELS = 6
PAINTED_RGB_CHANNELS = 3
GROUND_TO_VEHICLE_AXES = (
    (0., 1., 0.),
    (-1., 0., 0.),
    (0., 0., 1.),
)


class _SparseBasicBlock(spconv.SparseModule):
    """Two submanifold convolutions with a sparse residual connection."""

    def __init__(self, channels, indice_key):
        super().__init__()
        self.conv1 = spconv.SubMConv2d(
            channels, channels, 3, padding=1, bias=False,
            indice_key=indice_key)
        self.norm1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = spconv.SubMConv2d(
            channels, channels, 3, padding=1, bias=False,
            indice_key=indice_key)
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        identity = inputs.features
        output = self.conv1(inputs)
        output = output.replace_feature(
            self.relu(self.norm1(output.features)))
        output = self.conv2(output)
        output = output.replace_feature(
            self.relu(self.norm2(output.features) + identity))
        return output


class _SparseResNet2D(nn.Module):
    """PillarNeXt-style sparse ResNet with an overall stride of four."""

    def __init__(self, in_channels, base_channels, out_channels):
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4)
        norm = dict(eps=1e-3, momentum=0.01)
        self.stem = spconv.SparseSequential(
            spconv.SubMConv2d(
                in_channels, channels[0], 3, padding=1, bias=False,
                indice_key='pillar_stage0'),
            nn.BatchNorm1d(channels[0], **norm),
            nn.ReLU(inplace=True),
            _SparseBasicBlock(channels[0], 'pillar_stage0'),
            _SparseBasicBlock(channels[0], 'pillar_stage0'))
        self.stage1 = spconv.SparseSequential(
            spconv.SparseConv2d(
                channels[0], channels[1], 3, stride=2, padding=1,
                bias=False, indice_key='pillar_down1'),
            nn.BatchNorm1d(channels[1], **norm),
            nn.ReLU(inplace=True),
            _SparseBasicBlock(channels[1], 'pillar_stage1'),
            _SparseBasicBlock(channels[1], 'pillar_stage1'))
        self.stage2 = spconv.SparseSequential(
            spconv.SparseConv2d(
                channels[1], channels[2], 3, stride=2, padding=1,
                bias=False, indice_key='pillar_down2'),
            nn.BatchNorm1d(channels[2], **norm),
            nn.ReLU(inplace=True),
            _SparseBasicBlock(channels[2], 'pillar_stage2'),
            _SparseBasicBlock(channels[2], 'pillar_stage2'))
        self.mapping = spconv.SparseSequential(
            spconv.SubMConv2d(
                channels[2], out_channels, 1, bias=False,
                indice_key='pillar_mapping'),
            nn.BatchNorm1d(out_channels, **norm),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        return self.mapping(self.stage2(self.stage1(self.stem(inputs))))


class _DenseBasicBlock(nn.Module):
    """Depthwise-separable residual block before the dilated lane neck."""

    def __init__(self, channels):
        super().__init__()
        norm = dict(eps=1e-3, momentum=0.01)
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1,
            groups=channels, bias=False)
        self.norm1 = nn.BatchNorm2d(channels, **norm)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels, **norm)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.relu(self.norm1(self.conv1(inputs)))
        return self.relu(self.norm2(self.conv2(output)) + inputs)


class _ASPPNeck(nn.Module):
    """Memory-conscious ASPP after the sparse backbone is densified."""

    def __init__(self, channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        norm = dict(eps=1e-3, momentum=0.01)
        self.pre_conv = _DenseBasicBlock(channels)
        branch_channels = max(channels // 4, 16)
        self.point_branch = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels, **norm),
            nn.ReLU(inplace=True))
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3, padding=dilation,
                    dilation=dilation, groups=channels, bias=False),
                nn.BatchNorm2d(channels, **norm),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels, **norm),
                nn.ReLU(inplace=True))
            for dilation in dilations
        ])
        self.post_conv = nn.Sequential(
            nn.Conv2d(
                branch_channels * (len(dilations) + 1),
                channels, 1, bias=False),
            nn.BatchNorm2d(channels, **norm),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        inputs = self.pre_conv(inputs)
        branches = [self.point_branch(inputs)]
        branches.extend(branch(inputs) for branch in self.dilated_branches)
        return self.post_conv(torch.cat(branches, dim=1))


@MODELS.register_module()
class DynamicPillarBEVEncoder(nn.Module):
    """Encode accumulated points with dynamic PFE and a sparse lane network.

    Point features are learned before aggregation. The max/mean learned
    descriptors and deterministic road-surface statistics are concatenated
    once per non-empty pillar; statistics are never copied back to points.
    """

    point_input_channels = 15
    pillar_statistic_channels = 10
    statistic_channels = pillar_statistic_channels

    def __init__(self, point_cloud_range, pillar_size, out_channels=128,
                 queue_length=3, base_channels=64, **kwargs):
        super().__init__()
        self.point_cloud_range = tuple(float(value)
                                       for value in point_cloud_range)
        self.pillar_size = tuple(float(value) for value in pillar_size)
        self.out_channels = int(out_channels)
        self.queue_length = int(queue_length)
        self.pfe_channels = int(base_channels)
        self.pillar_feature_channels = (
            self.pfe_channels * 2 + self.pillar_statistic_channels)
        if len(self.point_cloud_range) != 6:
            raise ValueError('point_cloud_range must contain six values.')
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
        if self.grid_h % 4 or self.grid_w % 4:
            raise ValueError(
                'Dynamic pillar grid dimensions must be divisible by four.')

        norm = dict(eps=1e-3, momentum=0.01)
        self.point_mlp = nn.Sequential(
            nn.Linear(
                self.point_input_channels, self.pfe_channels, bias=False),
            nn.BatchNorm1d(self.pfe_channels, **norm),
            nn.ReLU(inplace=True),
            nn.Linear(self.pfe_channels, self.pfe_channels, bias=False),
            nn.BatchNorm1d(self.pfe_channels, **norm),
            nn.ReLU(inplace=True))
        self.pillar_projection = nn.Sequential(
            nn.Linear(
                self.pillar_feature_channels, base_channels, bias=False),
            nn.BatchNorm1d(base_channels, **norm),
            nn.ReLU(inplace=True))
        self.backbone = _SparseResNet2D(
            base_channels, base_channels, self.out_channels)
        self.lane_neck = _ASPPNeck(self.out_channels)

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        point_weight_key = prefix + 'point_mlp.0.weight'
        point_weight = state_dict.get(point_weight_key)
        expected_weight = self.point_mlp[0].weight
        if (point_weight is not None
                and point_weight.shape[0] == expected_weight.shape[0]
                and point_weight.shape[1] == (
                    expected_weight.shape[1] - PAINTED_RGB_CHANNELS)):
            expanded_weight = point_weight.new_zeros(expected_weight.shape)
            expanded_weight[:, :point_weight.shape[1]] = point_weight
            state_dict[point_weight_key] = expanded_weight
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _scatter_sum(values, indices, size):
        output = values.new_zeros((size,) + values.shape[1:])
        output.index_add_(0, indices, values)
        return output

    @staticmethod
    def _scatter_extreme(values, indices, size, reduce):
        fill = torch.inf if reduce == 'amin' else -torch.inf
        output = values.new_full((size,) + values.shape[1:], fill)
        expanded_indices = indices.view(
            -1, *([1] * (values.ndim - 1))).expand_as(values)
        output.scatter_reduce_(
            0, expanded_indices, values, reduce=reduce, include_self=True)
        return torch.where(torch.isfinite(output), output,
                           torch.zeros_like(output))

    def _valid_points(self, points):
        all_points = []
        all_flat_indices = []
        cells = self.grid_h * self.grid_w
        for batch_index, sample in enumerate(points):
            if sample.ndim != 2 or sample.size(-1) < 6:
                raise ValueError(
                    'Dynamic pillar points must contain '
                    'xyz/intensity/elongation/queue_index.')
            if sample.numel() == 0:
                continue
            if sample.size(-1) == BASE_POINT_CHANNELS:
                sample = torch.cat((
                    sample,
                    sample.new_zeros(sample.size(0), PAINTED_RGB_CHANNELS),
                ), dim=-1)
            elif sample.size(-1) < (
                    BASE_POINT_CHANNELS + PAINTED_RGB_CHANNELS):
                raise ValueError(
                    'Painted dynamic pillar points must append three RGB '
                    'channels after queue_index.')
            x_index = torch.floor(
                (sample[:, X] - self.point_cloud_range[0])
                / self.pillar_size[0]).long()
            y_index = torch.floor(
                (sample[:, Y] - self.point_cloud_range[1])
                / self.pillar_size[1]).long()
            valid = (
                (sample[:, X] >= self.point_cloud_range[0])
                & (sample[:, X] <= self.point_cloud_range[3])
                & (sample[:, Y] >= self.point_cloud_range[1])
                & (sample[:, Y] <= self.point_cloud_range[4])
                & (sample[:, Z] >= self.point_cloud_range[2])
                & (sample[:, Z] <= self.point_cloud_range[5]))
            if not valid.any():
                continue
            sample = sample[
                valid, :BASE_POINT_CHANNELS + PAINTED_RGB_CHANNELS]
            flat_index = (
                y_index[valid].clamp(0, self.grid_h - 1) * self.grid_w
                + x_index[valid].clamp(0, self.grid_w - 1)
                + batch_index * cells)
            all_points.append(sample)
            all_flat_indices.append(flat_index)
        if not all_points:
            reference = points[0]
            return (
                reference.new_zeros((
                    0, BASE_POINT_CHANNELS + PAINTED_RGB_CHANNELS)),
                torch.zeros(0, dtype=torch.long, device=reference.device))
        return torch.cat(all_points), torch.cat(all_flat_indices)

    @staticmethod
    def _frame_meta(img_metas, batch_index, time_index):
        if img_metas is None:
            return None
        sample_meta = (
            img_metas[batch_index]
            if isinstance(img_metas, (list, tuple)) else img_metas)
        if not isinstance(sample_meta, dict):
            return None
        queue = sample_meta.get('queue')
        if isinstance(queue, dict):
            return queue.get(time_index, queue.get(str(time_index)))
        if isinstance(queue, (list, tuple)) and time_index < len(queue):
            return queue[time_index]
        return sample_meta.get('current', sample_meta)

    def paint_points(self, points, images=None, img_metas=None):
        """Append normalized RGB sampled from each point's temporal image."""
        if not isinstance(points, (list, tuple)):
            points = [points]
        if images is not None and images.ndim == 5:
            images = images.unsqueeze(1)
        if images is not None and images.ndim != 6:
            raise ValueError(
                'Point painting images must have shape [B,T,N,3,H,W] or '
                f'[B,N,3,H,W], got {tuple(images.shape)}.')
        if images is not None and images.size(0) != len(points):
            raise ValueError(
                'Point painting image batch must match the point batch.')

        painted_batch = []
        for batch_index, sample in enumerate(points):
            if sample.ndim != 2 or sample.size(-1) < BASE_POINT_CHANNELS:
                raise ValueError(
                    'Dynamic pillar points must contain '
                    'xyz/intensity/elongation/queue_index.')
            base = sample[:, :BASE_POINT_CHANNELS]
            rgb = base.new_zeros(sample.size(0), PAINTED_RGB_CHANNELS)
            if images is None or img_metas is None or sample.numel() == 0:
                painted_batch.append(torch.cat((base, rgb), dim=-1))
                continue

            time_steps = images.size(1)
            queue_indices = base[:, QUEUE_INDEX].round().long()
            ground_to_vehicle = base.new_tensor(GROUND_TO_VEHICLE_AXES)
            for time_index in queue_indices.unique().tolist():
                if not 0 <= time_index < time_steps:
                    continue
                frame_meta = self._frame_meta(
                    img_metas, batch_index, time_index)
                if not isinstance(frame_meta, dict):
                    continue
                lidar_to_image_matrices = frame_meta.get('lidar2img')
                if (lidar_to_image_matrices is None
                        or len(lidar_to_image_matrices) == 0):
                    continue
                lidar_to_image = torch.as_tensor(
                    lidar_to_image_matrices[0],
                    device=sample.device, dtype=sample.dtype)
                history_to_current = torch.as_tensor(
                    frame_meta.get('lidaradj2lidarcurr', torch.eye(4)),
                    device=sample.device, dtype=sample.dtype)

                point_indices = torch.where(queue_indices == time_index)[0]
                homogeneous = torch.cat((
                    base[point_indices, XYZ],
                    base.new_ones(point_indices.numel(), 1),
                ), dim=-1)
                history_points = torch.linalg.solve(
                    history_to_current, homogeneous.transpose(0, 1)
                ).transpose(0, 1)
                vehicle_points = torch.cat((
                    history_points[:, :3] @ ground_to_vehicle.transpose(0, 1),
                    history_points[:, 3:],
                ), dim=-1)
                projected = vehicle_points @ lidar_to_image.transpose(0, 1)
                depth = projected[:, 2]
                normalized_image = images[batch_index, time_index, 0]
                height, width = normalized_image.shape[-2:]
                grid = torch.stack((
                    projected[:, 0] / depth.clamp_min(1e-5)
                    / max(width - 1, 1) * 2 - 1,
                    projected[:, 1] / depth.clamp_min(1e-5)
                    / max(height - 1, 1) * 2 - 1,
                ), dim=-1)
                valid = (
                    (depth > 1e-5)
                    & (grid[:, 0].abs() <= 1)
                    & (grid[:, 1].abs() <= 1))
                sampled = F.grid_sample(
                    normalized_image.unsqueeze(0), grid.view(1, 1, -1, 2),
                    align_corners=True).view(PAINTED_RGB_CHANNELS, -1).T
                rgb[point_indices] = torch.where(
                    valid.unsqueeze(-1), sampled, torch.zeros_like(sampled))
            painted_batch.append(torch.cat((base, rgb), dim=-1))
        return painted_batch

    def _pillar_statistics(
            self, points, inverse, pillar_count, count, denominator):
        z = points[:, Z]
        intensity = torch.log1p(points[:, INTENSITY].clamp_min(0))
        elongation = points[:, ELONGATION]
        queue = (
            points[:, QUEUE_INDEX] / max(self.queue_length - 1, 1))
        z_sum = self._scatter_sum(z, inverse, pillar_count)
        z_mean = z_sum / denominator
        z_std = (
            self._scatter_sum(z.square(), inverse, pillar_count)
            / denominator - z_mean.square()).clamp_min(0).sqrt()
        return torch.stack((
            torch.log1p(count),
            self._scatter_extreme(z, inverse, pillar_count, 'amin'),
            self._scatter_extreme(z, inverse, pillar_count, 'amax'),
            z_mean,
            z_std,
            self._scatter_sum(intensity, inverse, pillar_count) / denominator,
            self._scatter_extreme(
                intensity, inverse, pillar_count, 'amax'),
            self._scatter_sum(elongation, inverse, pillar_count) / denominator,
            self._scatter_extreme(
                elongation, inverse, pillar_count, 'amax'),
            self._scatter_sum(queue, inverse, pillar_count) / denominator,
        ), dim=-1)

    def pillarize(self, points):
        """Return one learned descriptor and statistic vector per pillar."""
        if not isinstance(points, (list, tuple)):
            points = [points]
        if not points:
            raise ValueError('Dynamic pillar encoder requires a point batch.')
        point_values, flat_indices = self._valid_points(points)
        if point_values.numel() == 0:
            return dict(
                features=point_values.new_zeros(
                    (0, self.pillar_feature_channels)),
                statistics=point_values.new_zeros(
                    (0, self.pillar_statistic_channels)),
                coordinates=torch.zeros(
                    (0, 3), dtype=torch.int32,
                    device=point_values.device),
                batch_size=len(points))

        unique_indices, inverse = torch.unique(
            flat_indices, sorted=True, return_inverse=True)
        pillar_count = unique_indices.numel()
        count = self._scatter_sum(
            point_values.new_ones(point_values.size(0)),
            inverse, pillar_count)
        denominator = count.clamp_min(1)
        xyz_mean = (
            self._scatter_sum(
                point_values[:, XYZ], inverse, pillar_count)
            / denominator.unsqueeze(-1))
        point_cluster_offset = (
            point_values[:, XYZ] - xyz_mean[inverse])

        cells = self.grid_h * self.grid_w
        cell_indices = flat_indices.remainder(cells)
        x_indices = cell_indices.remainder(self.grid_w)
        y_indices = torch.div(
            cell_indices, self.grid_w, rounding_mode='floor')
        pillar_centers = torch.stack((
            self.point_cloud_range[0]
            + (x_indices.to(point_values.dtype) + .5) * self.pillar_size[0],
            self.point_cloud_range[1]
            + (y_indices.to(point_values.dtype) + .5) * self.pillar_size[1],
            point_values.new_full(
                (point_values.size(0),),
                (self.point_cloud_range[2]
                 + self.point_cloud_range[5]) * .5),
        ), dim=-1)
        point_center_offset = point_values[:, XYZ] - pillar_centers
        normalized_raw = point_values.clone()
        normalized_raw[:, INTENSITY] = torch.log1p(
            normalized_raw[:, INTENSITY].clamp_min(0))
        normalized_raw[:, QUEUE_INDEX] = (
            normalized_raw[:, QUEUE_INDEX]
            / max(self.queue_length - 1, 1))
        point_inputs = torch.cat((
            normalized_raw[:, :BASE_POINT_CHANNELS],
            point_cluster_offset,
            point_center_offset,
            normalized_raw[:, BASE_POINT_CHANNELS:],
        ), dim=-1)
        learned_points = self.point_mlp(point_inputs)
        learned_max = self._scatter_extreme(
            learned_points, inverse, pillar_count, 'amax')
        learned_mean = (
            self._scatter_sum(learned_points, inverse, pillar_count)
            / denominator.unsqueeze(-1))
        statistics = self._pillar_statistics(
            point_values, inverse, pillar_count, count, denominator)

        batch_indices = torch.div(
            unique_indices, cells, rounding_mode='floor')
        unique_cells = unique_indices.remainder(cells)
        coordinates = torch.stack((
            batch_indices,
            torch.div(unique_cells, self.grid_w, rounding_mode='floor'),
            unique_cells.remainder(self.grid_w),
        ), dim=-1).to(torch.int32)
        return dict(
            features=torch.cat(
                (learned_max, learned_mean, statistics), dim=-1),
            statistics=statistics,
            coordinates=coordinates,
            batch_size=len(points))

    def rasterize(self, points):
        """Return the deterministic statistic raster for diagnostics."""
        pillars = self.pillarize(points)
        statistics = pillars['statistics']
        coordinates = pillars['coordinates'].long()
        raster = statistics.new_zeros(
            pillars['batch_size'], self.pillar_statistic_channels,
            self.grid_h, self.grid_w)
        if coordinates.numel():
            raster[
                coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2]
            ] = statistics
        return raster

    def _occupancy(self, coordinates, batch_size, device):
        occupancy = torch.zeros(
            batch_size, 1, self.grid_h, self.grid_w,
            dtype=torch.float32, device=device)
        if coordinates.numel():
            coordinates = coordinates.long()
            occupancy[
                coordinates[:, 0], 0,
                coordinates[:, 1], coordinates[:, 2]] = 1
        occupancy = F.max_pool2d(
            occupancy, kernel_size=3, stride=2, padding=1)
        occupancy = F.max_pool2d(
            occupancy, kernel_size=3, stride=2, padding=1)
        return F.interpolate(
            occupancy, size=(self.grid_h // 2, self.grid_w // 2),
            mode='nearest') > 0

    def forward(self, points, img_feats=None, img_metas=None, images=None):
        del img_feats
        pillars = self.pillarize(
            self.paint_points(points, images=images, img_metas=img_metas))
        batch_size = pillars['batch_size']
        occupancy = self._occupancy(
            pillars['coordinates'], batch_size, pillars['features'].device)
        output_size = (self.grid_h // 2, self.grid_w // 2)
        if not pillars['features'].numel():
            return dict(
                features=pillars['features'].new_zeros(
                    batch_size, self.out_channels, *output_size),
                occupancy=occupancy)

        sparse_inputs = spconv.SparseConvTensor(
            self.pillar_projection(pillars['features']),
            pillars['coordinates'],
            spatial_shape=(self.grid_h, self.grid_w),
            batch_size=batch_size)
        dense_features = self.backbone(sparse_inputs).dense()
        dense_features = self.lane_neck(dense_features)
        features = F.interpolate(
            dense_features, size=output_size,
            mode='bilinear', align_corners=False)
        sample_valid = occupancy.flatten(1).any(dim=-1)
        features = features * sample_valid[:, None, None, None].to(
            features.dtype)
        return dict(features=features, occupancy=occupancy)
