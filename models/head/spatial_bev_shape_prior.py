"""Read ordered road evidence before compressing BEV into curve parameters."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from .lidar_shape_token_prior import LiDARShapeTokenPrior


class _RoadContextBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 3), padding=(3 * dilation, 1),
                      dilation=(dilation, 1), groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU())

    def forward(self, x):
        return x + self.context(x)


@MODELS.register_module()
class SpatialBEVShapePrior(LiDARShapeTokenPrior):
    """Extract five relative cubic templates from spatial BEV evidence.

    Preserve lateral resolution, read each longitudinal row with separate
    template queries, then connect the ordered row evidence bidirectionally.
    Three control queries per template retrieve near/middle/far evidence before
    producing exactly the existing three x/z slopes. Neither the curve family
    nor the lateral seed lattice changes. Inputs are BEV and occupancy only.
    """

    def __init__(self, longitudinal_rows=50, spatial_layers=2, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        channels = self.shape_tokens.embedding_dim
        if longitudinal_rows < 3:
            raise ValueError('longitudinal_rows must be at least three')
        if channels % 8 or channels % num_heads:
            raise ValueError('hidden_channels must be divisible by 8 and num_heads')
        self.longitudinal_rows = int(longitudinal_rows)
        # Replace the global 4x4 -> shared-vector path, rather than adding a
        # residual trained to compensate for it on particular examples.
        in_channels = self.feature_encoder[0].in_channels
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels), nn.GELU(),
            _RoadContextBlock(channels, 1), _RoadContextBlock(channels, 3))
        del self.template_head
        self.position_encoder = nn.Sequential(nn.Linear(2, channels), nn.GELU(),
                                               nn.Linear(channels, channels))
        self.row_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.row_norm = nn.LayerNorm(channels)
        layer = nn.TransformerEncoderLayer(channels, num_heads, channels * 4,
                                           dropout=0., batch_first=True, norm_first=True)
        self.longitudinal_encoder = nn.TransformerEncoder(layer, spatial_layers,
                                                          enable_nested_tensor=False)
        self.control_queries = nn.Embedding(3, channels)
        self.control_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.control_norm = nn.LayerNorm(channels)
        self.slope_head = nn.Linear(channels, 2)
        nn.init.normal_(self.control_queries.weight, std=.02)
        nn.init.xavier_uniform_(self.slope_head.weight, gain=.05)
        nn.init.zeros_(self.slope_head.bias)

    def _ordered_evidence(self, pts_feats):
        feature = self._feature(pts_feats)
        encoded = self.feature_encoder(feature)
        memory, observed = self._row_memory(encoded, pts_feats)
        return self._read_rows(memory, observed), feature, observed

    def _row_memory(self, encoded, pts_feats):
        batch, channels, _, width = encoded.shape
        rows = self.longitudinal_rows
        # Only pool along y. A 0.2 m lateral cell remains a 0.2 m cell; no 4x4
        # compression merges neighboring lane/curb structures across x.
        memory = F.adaptive_avg_pool2d(encoded, (rows, width))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, rows, device=encoded.device, dtype=encoded.dtype),
                                torch.linspace(-1, 1, width, device=encoded.device, dtype=encoded.dtype),
                                indexing='ij')
        position = self.position_encoder(torch.stack((xx, yy), -1))
        memory = memory.permute(0, 2, 3, 1) + position[None]
        occupancy = pts_feats.get('occupancy') if isinstance(pts_feats, dict) else None
        if occupancy is None:
            observed = torch.ones(batch, rows, width, dtype=torch.bool, device=encoded.device)
        else:
            observed = F.adaptive_max_pool2d(occupancy.float(), (rows, width))[:, 0] > 0
        return memory, observed

    def _read_rows(self, memory, observed, attention_bias=None, return_weights=False):
        batch, rows, width, channels = memory.shape
        yy = torch.linspace(-1, 1, rows, device=memory.device, dtype=memory.dtype)
        memory = memory.reshape(batch * rows, width, channels)
        # Null evidence makes completely empty rows valid attention inputs.
        memory = torch.cat((memory, memory.new_zeros(batch * rows, 1, channels)), 1)
        if attention_bias is None:
            padding = torch.cat((~observed.reshape(batch * rows, width),
                                 observed.new_zeros(batch * rows, 1)), 1)
        else:
            # A predicted lane can lie between returns. Do not force its
            # semantic position to coincide with an occupied LiDAR cell.
            padding = None
            if attention_bias.ndim == 3:
                attention_bias = F.pad(attention_bias.reshape(batch * rows, width), (0, 1))
                attention_bias = attention_bias[:, None].expand(-1, self.max_templates, -1)
            else:
                if attention_bias.shape != (batch, rows, self.max_templates, width):
                    raise ValueError('Per-template row attention bias has an incompatible shape')
                attention_bias = F.pad(attention_bias.reshape(batch * rows, self.max_templates, width), (0, 1))
            attention_bias = attention_bias.repeat_interleave(self.row_attention.num_heads, 0)
        query = (self.shape_tokens.weight[None, None]
                 + self.position_encoder(torch.stack((yy * 0, yy), -1))[None, :, None])
        query = query.expand(batch, -1, -1, -1).reshape(batch * rows, self.max_templates, channels)
        attended, weights = self.row_attention(query, memory, memory,
                                               key_padding_mask=padding,
                                               attn_mask=attention_bias, need_weights=return_weights)
        evidence = self.row_norm(query + attended)
        evidence = evidence.reshape(batch, rows, self.max_templates, channels).permute(0, 2, 1, 3)
        evidence = self.longitudinal_encoder(evidence.reshape(batch * self.max_templates, rows, channels))
        if return_weights:
            return evidence, weights.reshape(batch, rows, self.max_templates, width+1)
        return evidence

    def _relative_controls(self, pts_feats, img_feats=None):
        if pts_feats is None:
            raise ValueError('SpatialBEVShapePrior requires LiDAR BEV features')
        evidence, feature, observed = self._ordered_evidence(pts_feats)
        return self._controls_from_evidence(evidence, feature, observed)

    def _controls_from_evidence(self, evidence, feature, observed):
        batch, channels = feature.size(0), evidence.size(-1)
        # Explicit longitudinal locations anchor the slope/control queries.
        y = feature.new_tensor([-2 / 3, 0., 2 / 3])
        position = self.position_encoder(torch.stack((y * 0, y), -1))
        query = self.control_queries.weight + position
        query = query[None].expand(batch * self.max_templates, -1, -1)
        controls = self.control_norm(query + self.control_attention(
            query, evidence, evidence, need_weights=False)[0])
        raw = self.slope_head(controls).reshape(batch, self.max_templates, 3, 2)
        valid = observed.flatten(1).any(-1)
        raw = raw * valid[:, None, None, None].to(raw)
        lateral = raw[..., 0] * self.lateral_slope_scale
        vertical = raw[..., 1] * self.vertical_slope_scale
        length = self.anchor_y_steps[-1].to(raw) / 3
        zeros = raw.new_zeros(batch, self.max_templates, 1)
        x = torch.cat((zeros, (lateral * length).cumsum(-1)), -1)
        z = torch.cat((zeros, (vertical * length).cumsum(-1)), -1)
        tokens = evidence.mean(1).reshape(batch, self.max_templates, channels)
        tokens = tokens * valid[:, None, None].to(tokens)
        return (x, z, lateral, vertical, torch.rad2deg(torch.atan(lateral)),
                torch.rad2deg(torch.atan(vertical)), tokens, valid)
