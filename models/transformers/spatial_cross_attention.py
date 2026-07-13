"""Image-space cross attention for BEV queries."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from .bev_grid import bev_cell_centers


@MODELS.register_module()
class LaneFormerSpatialCrossAttention(nn.Module):
    """Project 3D BEV reference points and sample image features at each view."""

    def __init__(self, embed_dims, bev_h, bev_w, pc_range,
                 num_points_in_pillar=4, reference_zs=None, num_levels=1,
                 **kwargs):
        super().__init__()
        if num_levels != 1:
            raise ValueError('LaneFormer currently receives one image feature level.')
        self.embed_dims = embed_dims
        self.bev_h, self.bev_w = bev_h, bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        if reference_zs is not None and len(reference_zs) != num_points_in_pillar:
            raise ValueError('reference_zs length must equal num_points_in_pillar.')
        if reference_zs is not None and (
                min(reference_zs) < pc_range[2] or max(reference_zs) > pc_range[5]):
            raise ValueError('reference_zs must lie inside point_cloud_range z bounds.')
        self.register_buffer(
            'reference_zs', torch.tensor(
                reference_zs if reference_zs is not None else [], dtype=torch.float32),
            persistent=False)
        self.height_attention = nn.Conv2d(embed_dims, num_points_in_pillar, 1)
        self.output_proj = nn.Conv2d(embed_dims, embed_dims, 1)

    @staticmethod
    def _meta(metas, index):
        if metas is None:
            return {}
        meta = metas[index] if isinstance(metas, (list, tuple)) else metas
        return meta.get('current', meta) if isinstance(meta, dict) else {}

    def _reference_points(self, device, dtype):
        y, x = bev_cell_centers(
            self.pc_range, self.bev_h, self.bev_w, device, dtype)
        z = (self.reference_zs.to(device=device, dtype=dtype)
             if self.reference_zs.numel()
             else torch.linspace(self.pc_range[2], self.pc_range[5],
                                 self.num_points_in_pillar, device=device, dtype=dtype))
        points = torch.stack((
            x.expand(self.num_points_in_pillar, -1, -1),
            y.expand(self.num_points_in_pillar, -1, -1),
            z[:, None, None].expand(-1, self.bev_h, self.bev_w),
            torch.ones((self.num_points_in_pillar, self.bev_h, self.bev_w),
                       device=device, dtype=dtype)), dim=-1)
        return points.reshape(self.num_points_in_pillar, -1, 4).transpose(1, 2)

    def forward(self, img_feats, img_metas, bev_query):
        feature = img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats
        batch, cameras, _, feat_h, feat_w = feature.shape
        references = self._reference_points(feature.device, feature.dtype)
        height_weights = self.height_attention(bev_query).softmax(dim=1)
        output = feature.new_zeros((batch, self.embed_dims, self.bev_h, self.bev_w))
        for batch_idx in range(batch):
            meta = self._meta(img_metas, batch_idx)
            matrices = meta.get('lidar2img', [])
            image_shape = meta.get('img_shape', (feat_h, feat_w))
            image_h, image_w = float(image_shape[0]), float(image_shape[1])
            samples, weights = [], []
            for camera_idx in range(cameras):
                matrix = matrices[min(camera_idx, len(matrices) - 1)] if matrices else torch.eye(4)
                matrix = torch.as_tensor(matrix, device=feature.device, dtype=feature.dtype)
                for point_idx in range(self.num_points_in_pillar):
                    projected = matrix @ references[point_idx]
                    depth = projected[2]
                    valid_depth = depth > 1e-5
                    depth = depth.clamp_min(1e-5)
                    grid = torch.stack((
                        projected[0] / depth / max(image_w - 1., 1.) * 2 - 1,
                        projected[1] / depth / max(image_h - 1., 1.) * 2 - 1), dim=-1)
                    valid = valid_depth & (grid[:, 0].abs() <= 1) & (grid[:, 1].abs() <= 1)
                    sample = F.grid_sample(
                        feature[batch_idx, camera_idx:camera_idx + 1],
                        grid.reshape(1, self.bev_h, self.bev_w, 2),
                        align_corners=True)
                    weight = valid.reshape(1, 1, self.bev_h, self.bev_w).to(feature.dtype)
                    weight = weight * height_weights[batch_idx:batch_idx + 1, point_idx:point_idx + 1]
                    samples.append(sample * weight)
                    weights.append(weight)
            if samples:
                output[batch_idx] = torch.cat(samples, dim=0).sum(0) / torch.cat(
                    weights, dim=0).sum(0).clamp_min(1.)
        return self.output_proj(output)
