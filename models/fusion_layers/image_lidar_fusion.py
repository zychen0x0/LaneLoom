"""Image/LiDAR BEV fusion at each BEV query location."""
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS


class _BEVModalityEncoderLayer(nn.Module):
    """A BEV query attends to its image and LiDAR feature tokens."""

    def __init__(self, embed_dims, num_heads, feedforward_channels, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims))
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, query, image_bev, lidar_bev):
        batch, channels, height, width = query.shape
        query_tokens = query.flatten(2).transpose(1, 2).reshape(-1, 1, channels)
        tokens = torch.stack((image_bev, lidar_bev), dim=2)
        tokens = tokens.permute(0, 3, 4, 2, 1).reshape(-1, 2, channels)
        attended = self.cross_attn(
            query_tokens, tokens, tokens, need_weights=False)[0]
        query_tokens = self.norm1(query_tokens + attended)
        query_tokens = self.norm2(query_tokens + self.ffn(query_tokens))
        return query_tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


@MODELS.register_module()
class LaneLoomImageLiDARFusion(nn.Module):
    """Cross-attend BEV queries to image and LiDAR tokens at each BEV cell."""

    def __init__(self, embed_dims, num_layers=2,
                 num_heads=8, feedforward_channels=1024, dropout=0.1,
                 fusion_method='cross_attention', **kwargs):
        super().__init__()
        if fusion_method != 'cross_attention':
            raise ValueError('Only cross_attention fusion is implemented.')
        self.layers = nn.ModuleList([
            _BEVModalityEncoderLayer(
                embed_dims, num_heads, feedforward_channels, dropout)
            for _ in range(num_layers)])

    def forward(self, image_bev, lidar_bev, bev_query):
        output = bev_query
        for layer in self.layers:
            output = layer(output, image_bev, lidar_bev)
        return output
