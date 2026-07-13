"""BEV query encoder composed from spatial, fusion, and temporal modules."""
import torch.nn as nn
from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerBEVEncoder(nn.Module):
    """Expose current-frame BEV encoding and historical BEV fusion."""

    def __init__(self, embed_dims, bev_h, bev_w, pc_range,
                 spatial_cross_attn, lidar_fusion, temporal_self_attn,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h, self.bev_w = bev_h, bev_w
        self.bev_query = nn.Embedding(bev_h * bev_w, embed_dims)
        self.bev_pos = nn.Embedding(bev_h * bev_w, embed_dims)
        self.spatial_cross_attn = MODELS.build(spatial_cross_attn)
        self.lidar_fusion = MODELS.build(lidar_fusion)
        self.temporal_self_attn = MODELS.build(temporal_self_attn)

    def _query_map(self, batch_size):
        query = (self.bev_query.weight + self.bev_pos.weight).transpose(0, 1)
        query = query.reshape(1, self.embed_dims, self.bev_h, self.bev_w)
        return query.expand(batch_size, -1, -1, -1)

    def forward_bev(self, img_feats, pts_feats, img_metas):
        batch_size = img_feats[0].size(0) if isinstance(img_feats, (list, tuple)) else img_feats.size(0)
        query = self._query_map(batch_size)
        image_bev = self.spatial_cross_attn(img_feats, img_metas, query)
        return self.lidar_fusion(image_bev, pts_feats, query)

    def forward(self, img_feats, pts_feats, img_metas, prev_bev=None):
        return self.temporal_self_attn(
            self.forward_bev(img_feats, pts_feats, img_metas), prev_bev)
