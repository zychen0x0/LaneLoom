"""LiDAR BEV adapter used by the curve-query decoder."""
import torch.nn as nn
from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVEncoder(nn.Module):
    """Project accumulated LiDAR features onto the lane-query BEV grid."""

    def __init__(self, bev_h, bev_w, lidar_bev_adapter):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.lidar_bev_adapter = MODELS.build(lidar_bev_adapter)

    def forward_image_bev(self, img_feats, img_metas):
        return None

    def forward_lidar_bev(self, pts_feats):
        if pts_feats is None:
            return None
        return self.lidar_bev_adapter(
            pts_feats, output_size=(self.bev_h, self.bev_w))

    def forward_bev(self, img_feats, pts_feats, img_metas):
        return self.forward_lidar_bev(pts_feats)

    def forward(self, img_feats, pts_feats, img_metas, prev_bev=None):
        return self.forward_bev(img_feats, pts_feats, img_metas)
