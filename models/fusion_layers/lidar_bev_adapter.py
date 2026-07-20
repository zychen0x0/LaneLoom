"""LiDAR feature adaptation into the shared BEV representation."""

import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class LiDARBEVAdapter(nn.Module):
    """Project the LiDAR backbone output onto LaneLoom's BEV grid."""

    def __init__(self, embed_dims, lidar_in_channels, **kwargs):
        super().__init__()
        self.projection = (
            nn.Identity() if lidar_in_channels == embed_dims
            else nn.Conv2d(lidar_in_channels, embed_dims, 1))

    def forward(self, pts_feats, output_size):
        if pts_feats is None:
            return None
        if isinstance(pts_feats, dict):
            pts_feats = pts_feats['features']
        feature = pts_feats[-1] if isinstance(pts_feats, (list, tuple)) else pts_feats
        if tuple(feature.shape[-2:]) != tuple(output_size):
            feature = F.interpolate(
                feature, output_size, mode='bilinear', align_corners=False)
        return self.projection(feature)
