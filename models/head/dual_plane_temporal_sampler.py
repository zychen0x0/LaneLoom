"""Query-aligned deformable temporal FV and LiDAR-BEV sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class DualPlaneTemporalSampler(nn.Module):
    """Build local deformable evidence around every 3D curve sample."""

    def __init__(self, image_in_channels, lidar_in_channels, embed_dims,
                 pc_range, num_image_levels=1, temporal_kernel_size=3,
                 num_image_offsets=4, num_lidar_offsets=4,
                 max_image_offset_pixels=2.,
                 max_lidar_offset_residual=.5, **kwargs):
        super().__init__()
        self.pc_range = tuple(float(value) for value in pc_range)
        self.embed_dims = int(embed_dims)
        self.num_image_levels = int(num_image_levels)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.num_image_offsets = int(num_image_offsets)
        self.num_lidar_offsets = int(num_lidar_offsets)
        self.max_image_offset_pixels = float(max_image_offset_pixels)
        self.max_lidar_offset_residual = float(max_lidar_offset_residual)
        if self.temporal_kernel_size <= 0:
            raise ValueError('temporal_kernel_size must be positive.')
        if min(self.num_image_offsets, self.num_lidar_offsets) <= 0:
            raise ValueError('Deformable offset counts must be positive.')

        self.image_adapter = nn.Linear(image_in_channels, embed_dims)
        self.lidar_adapter = nn.Linear(lidar_in_channels, embed_dims)
        self.temporal_depthwise = nn.Conv1d(
            embed_dims, embed_dims, self.temporal_kernel_size,
            groups=embed_dims, bias=False)
        self.temporal_pointwise = nn.Conv1d(embed_dims, embed_dims, 1)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.image_offset_head = nn.Linear(
            embed_dims, self.num_image_offsets * 2)
        self.image_weight_head = nn.Linear(
            embed_dims, self.num_image_offsets)
        self.lidar_offset_head = nn.Linear(
            embed_dims, self.num_lidar_offsets)
        self.lidar_weight_head = nn.Linear(
            embed_dims, self.num_lidar_offsets)
        self.modality_embedding = nn.Embedding(2, embed_dims)
        self.level_embedding = nn.Embedding(self.num_image_levels, embed_dims)

        image_x = torch.linspace(
            -self.max_image_offset_pixels,
            self.max_image_offset_pixels,
            self.num_image_offsets)
        self.register_buffer(
            'image_base_offsets',
            torch.stack((image_x, torch.zeros_like(image_x)), dim=-1),
            persistent=False)
        lidar_base_offsets = (
            torch.tensor([-1., -.5, .5, 1.])
            if self.num_lidar_offsets == 4
            else torch.linspace(-1., 1., self.num_lidar_offsets))
        self.register_buffer(
            'lidar_base_offsets', lidar_base_offsets, persistent=False)
        for head in (
                self.image_offset_head, self.image_weight_head,
                self.lidar_offset_head, self.lidar_weight_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.temporal_pointwise.weight)
        nn.init.zeros_(self.temporal_pointwise.bias)

    @staticmethod
    def _sample_meta(img_metas, batch_idx):
        if img_metas is None:
            return {}
        return (img_metas[batch_idx]
                if isinstance(img_metas, (list, tuple)) else img_metas)

    @classmethod
    def _frame_meta(cls, img_metas, batch_idx, time_idx):
        sample = cls._sample_meta(img_metas, batch_idx)
        if not isinstance(sample, dict):
            return {}
        queue = sample.get('queue')
        if isinstance(queue, dict):
            return queue.get(time_idx, queue.get(str(time_idx), sample.get(
                'current', sample)))
        if isinstance(queue, (list, tuple)) and time_idx < len(queue):
            return queue[time_idx]
        return sample.get('current', sample)

    @staticmethod
    def _as_temporal_feature(feature):
        if feature.ndim == 5:
            return feature.unsqueeze(1)
        if feature.ndim != 6:
            raise ValueError(
                'Image feature levels must have shape [B,T,N,C,H,W] or '
                f'[B,N,C,H,W], got {tuple(feature.shape)}.')
        return feature

    def _point_position(self, curves):
        minimum = curves.new_tensor(self.pc_range[:3])
        maximum = curves.new_tensor(self.pc_range[3:])
        normalized = ((curves - minimum) / (maximum - minimum)).clamp(0, 1)
        return self.point_encoder(normalized)

    def _offset_context(self, curves, query_features):
        return self._point_position(curves) + query_features.unsqueeze(2)

    @staticmethod
    def _masked_attention(logits, valid):
        weights = torch.softmax(logits.masked_fill(~valid, -1e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _image_offsets(self, context):
        learned = torch.tanh(self.image_offset_head(context)).view(
            *context.shape[:-1], self.num_image_offsets, 2)
        learned = learned * self.max_image_offset_pixels
        return learned + self.image_base_offsets.to(context)

    def _sample_image(self, curves, query_features, image_features, img_metas):
        batch, queries, anchors, _ = curves.shape
        if image_features is None:
            empty = curves.new_zeros(
                batch, queries, anchors, self.num_image_levels, 1,
                self.image_adapter.in_features)
            valid = torch.zeros(
                batch, queries, anchors, self.num_image_levels, 1,
                dtype=torch.bool, device=curves.device)
            return empty, valid
        if len(image_features) != self.num_image_levels:
            raise ValueError(
                f'Expected {self.num_image_levels} image levels, got '
                f'{len(image_features)}.')

        levels = [self._as_temporal_feature(level)
                  for level in image_features]
        time_steps = levels[0].size(1)
        channels = levels[0].size(3)
        sampled_sum = curves.new_zeros(
            batch, queries, anchors, self.num_image_levels, time_steps,
            channels)
        weight_sum = curves.new_zeros(
            batch, queries, anchors, self.num_image_levels, time_steps, 1)
        homogeneous = torch.cat(
            (curves, torch.ones_like(curves[..., :1])), dim=-1)
        context = self._offset_context(curves, query_features)
        offsets = self._image_offsets(context)
        offset_logits = self.image_weight_head(context)

        for batch_idx in range(batch):
            for time_idx in range(time_steps):
                meta = self._frame_meta(img_metas, batch_idx, time_idx)
                history_to_current = torch.as_tensor(
                    meta.get('lidaradj2lidarcurr', torch.eye(4)),
                    device=curves.device, dtype=curves.dtype)
                history_points = torch.linalg.solve(
                    history_to_current,
                    homogeneous[batch_idx].reshape(-1, 4).transpose(0, 1))
                history_points = history_points.transpose(0, 1).reshape(
                    queries, anchors, 4)
                matrices = meta.get('lane2img') or meta.get('lidar2img', [])
                image_shape = meta.get('img_shape', levels[0].shape[-2:])
                image_h, image_w = float(image_shape[0]), float(image_shape[1])

                for level_idx, level in enumerate(levels):
                    feat_h, feat_w = level.shape[-2:]
                    normalized_offsets = torch.stack((
                        offsets[batch_idx, ..., 0]
                        * 2 / max(feat_w - 1, 1),
                        offsets[batch_idx, ..., 1]
                        * 2 / max(feat_h - 1, 1)), dim=-1)
                    cameras = level.size(2)
                    for camera_idx in range(cameras):
                        matrix = (
                            matrices[min(camera_idx, len(matrices) - 1)]
                            if matrices else torch.eye(4))
                        matrix = torch.as_tensor(
                            matrix, device=curves.device, dtype=curves.dtype)
                        projected = history_points @ matrix.transpose(0, 1)
                        depth = projected[..., 2]
                        base_grid = torch.stack((
                            projected[..., 0] / depth.clamp_min(1e-5)
                            / max(image_w - 1, 1) * 2 - 1,
                            projected[..., 1] / depth.clamp_min(1e-5)
                            / max(image_h - 1, 1) * 2 - 1), dim=-1)
                        grid = base_grid.unsqueeze(-2) + normalized_offsets
                        valid = (
                            (depth > 1e-5).unsqueeze(-1)
                            & (grid[..., 0].abs() <= 1)
                            & (grid[..., 1].abs() <= 1))
                        sampled = F.grid_sample(
                            level[
                                batch_idx, time_idx,
                                camera_idx].unsqueeze(0),
                            grid.reshape(
                                1, queries,
                                anchors * self.num_image_offsets, 2),
                            align_corners=True)
                        sampled = sampled.squeeze(0).permute(
                            1, 2, 0).reshape(
                                queries, anchors,
                                self.num_image_offsets, channels)
                        attention = self._masked_attention(
                            offset_logits[batch_idx], valid)
                        aggregated = (
                            sampled * attention.unsqueeze(-1)).sum(dim=-2)
                        camera_valid = valid.any(dim=-1).unsqueeze(-1)
                        sampled_sum[
                            batch_idx, :, :, level_idx, time_idx] += (
                                aggregated * camera_valid)
                        weight_sum[
                            batch_idx, :, :, level_idx, time_idx] += (
                                camera_valid)

        sampled = sampled_sum / weight_sum.clamp_min(1)
        return sampled, weight_sum.squeeze(-1) > 0

    def _fuse_time(self, sampled_image, valid_mask):
        projected = self.image_adapter(sampled_image)
        projected = projected * valid_mask.unsqueeze(-1).to(projected.dtype)
        batch, queries, anchors, levels, time_steps, channels = projected.shape
        sequence = projected.reshape(
            -1, time_steps, channels).transpose(1, 2)
        sequence = F.pad(
            sequence, (self.temporal_kernel_size - 1, 0))
        temporal = self.temporal_pointwise(
            self.temporal_depthwise(sequence))[:, :, -1]
        current = projected[..., -1, :].reshape(-1, channels)
        return (current + temporal).reshape(
            batch, queries, anchors, levels, channels)

    @staticmethod
    def _curve_normals(curves):
        xy = curves[..., :2]
        tangent = torch.empty_like(xy)
        tangent[..., 0, :] = xy[..., 1, :] - xy[..., 0, :]
        tangent[..., -1, :] = xy[..., -1, :] - xy[..., -2, :]
        if curves.size(-2) > 2:
            tangent[..., 1:-1, :] = (
                xy[..., 2:, :] - xy[..., :-2, :])
        normal = torch.stack(
            (tangent[..., 1], -tangent[..., 0]), dim=-1)
        return normal / torch.linalg.vector_norm(
            normal, dim=-1, keepdim=True).clamp_min(1e-6)

    def _sample_lidar(self, curves, query_features, lidar_bev,
                      lidar_occupancy=None):
        batch, queries, anchors, _ = curves.shape
        if lidar_bev is None:
            sampled = curves.new_zeros(
                batch, queries, anchors, self.lidar_adapter.in_features)
            valid = torch.zeros(
                batch, queries, anchors, dtype=torch.bool,
                device=curves.device)
            return sampled, valid
        context = self._offset_context(curves, query_features)
        residual = torch.tanh(
            self.lidar_offset_head(context)) * self.max_lidar_offset_residual
        distances = residual + self.lidar_base_offsets.to(context)
        normal = self._curve_normals(curves)
        sampled_xy = (
            curves[..., :2].unsqueeze(-2)
            + normal.unsqueeze(-2) * distances.unsqueeze(-1))
        grid_x = (
            (sampled_xy[..., 0] - self.pc_range[0])
            / (self.pc_range[3] - self.pc_range[0]) * 2 - 1)
        grid_y = (
            (sampled_xy[..., 1] - self.pc_range[1])
            / (self.pc_range[4] - self.pc_range[1]) * 2 - 1)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        valid = (grid_x.abs() <= 1) & (grid_y.abs() <= 1)
        flattened_grid = grid.reshape(
            batch, queries, anchors * self.num_lidar_offsets, 2)
        if lidar_occupancy is not None:
            if lidar_occupancy.ndim == 3:
                lidar_occupancy = lidar_occupancy.unsqueeze(1)
            sampled_occupancy = F.grid_sample(
                lidar_occupancy.to(dtype=curves.dtype),
                flattened_grid, mode='nearest',
                align_corners=False).squeeze(1).reshape(
                    batch, queries, anchors, self.num_lidar_offsets)
            valid &= sampled_occupancy > .5
        channels = lidar_bev.size(1)
        sampled = F.grid_sample(
            lidar_bev,
            flattened_grid,
            align_corners=False).permute(0, 2, 3, 1).reshape(
                batch, queries, anchors,
                self.num_lidar_offsets, channels)
        attention = self._masked_attention(
            self.lidar_weight_head(context), valid)
        sampled = (sampled * attention.unsqueeze(-1)).sum(dim=-2)
        return sampled, valid.any(dim=-1)

    def forward(self, curves, query_features, image_features,
                lidar_bev, img_metas, lidar_occupancy=None):
        if query_features is None:
            query_features = curves.new_zeros(
                curves.size(0), curves.size(1), self.embed_dims)
        sampled_image, image_valid = self._sample_image(
            curves, query_features, image_features, img_metas)
        image_tokens = self._fuse_time(sampled_image, image_valid)
        lidar_values, lidar_valid = self._sample_lidar(
            curves, query_features, lidar_bev, lidar_occupancy)
        lidar_tokens = self.lidar_adapter(lidar_values)
        position = self._point_position(curves)

        image_tokens = (
            image_tokens + position.unsqueeze(3)
            + self.modality_embedding.weight[0].view(1, 1, 1, 1, -1)
            + self.level_embedding.weight.view(
                1, 1, 1, self.num_image_levels, -1))
        lidar_tokens = (
            lidar_tokens + position
            + self.modality_embedding.weight[1].view(1, 1, 1, -1))
        image_token_valid = image_valid.any(dim=-1)
        batch, queries, anchors, levels, channels = image_tokens.shape
        tokens = torch.cat((
            image_tokens.reshape(
                batch, queries, anchors * levels, channels),
            lidar_tokens), dim=2)
        key_padding_mask = torch.cat((
            ~image_token_valid.reshape(
                batch, queries, anchors * levels),
            ~lidar_valid), dim=2)
        return dict(
            tokens=tokens,
            key_padding_mask=key_padding_mask,
            image_valid_mask=image_valid,
            lidar_valid_mask=lidar_valid)
