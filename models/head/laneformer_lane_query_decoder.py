"""Configurable lane-query decoder."""
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LaneFormerLaneQueryDecoder(nn.Module):
    """Decode BEV memory into cubic-Bézier 3D lanes sampled at anchors."""

    def __init__(self, embed_dims, bev_h, bev_w, pc_range, num_query,
                 anchor_y_steps, num_layers=3, num_heads=8,
                 feedforward_channels=1024, dropout=0.1, num_category=21,
                 **kwargs):
        super().__init__()
        self.pc_range = pc_range
        self.num_query = num_query
        self.register_buffer('anchor_y_steps', torch.tensor(anchor_y_steps, dtype=torch.float32))
        self.num_anchors = len(anchor_y_steps)
        self.num_control_points = 4
        anchor_span = self.anchor_y_steps[-1] - self.anchor_y_steps[0]
        if anchor_span <= 0:
            raise ValueError('anchor_y_steps must be strictly increasing.')
        t = (self.anchor_y_steps - self.anchor_y_steps[0]) / anchor_span
        self.register_buffer(
            'bezier_basis',
            torch.stack((
                (1 - t) ** 3,
                3 * (1 - t) ** 2 * t,
                3 * (1 - t) * t ** 2,
                t ** 3,
            ), dim=-1))
        self.num_category = int(num_category)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dims, nhead=num_heads,
            dim_feedforward=feedforward_channels, dropout=dropout,
            activation='relu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.query = nn.Embedding(num_query, embed_dims)
        self.query_pos = nn.Embedding(num_query, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)
        self.x_head = nn.Linear(embed_dims, self.num_control_points)
        self.z_head = nn.Linear(embed_dims, self.num_control_points)
        self.vis_head = nn.Linear(embed_dims, self.num_anchors)
        self.cls_head = nn.Linear(embed_dims, 1)
        self.category_head = nn.Linear(embed_dims, self.num_category)

    def forward(self, bev, bev_pos):
        batch = bev.size(0)
        memory = bev.flatten(2).transpose(1, 2)
        memory = memory + bev_pos.weight.unsqueeze(0).expand(batch, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(batch, -1, -1)
        query = self.query.weight.unsqueeze(0).expand(batch, -1, -1)
        query = self.norm(self.decoder(query + query_pos, memory))
        curve_x_controls = torch.sigmoid(self.x_head(query)) * (
            self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        curve_z_controls = torch.tanh(self.z_head(query)) * max(
            abs(self.pc_range[2]), abs(self.pc_range[5]))
        pred_x = torch.einsum(
            'ac,bqc->bqa', self.bezier_basis, curve_x_controls)
        pred_z = torch.einsum(
            'ac,bqc->bqa', self.bezier_basis, curve_z_controls)
        return dict(
            pred_x=pred_x, pred_z=pred_z,
            curve_x_controls=curve_x_controls,
            curve_z_controls=curve_z_controls,
            pred_vis=self.vis_head(query), pred_cls=self.cls_head(query).squeeze(-1),
            pred_category=self.category_head(query),
            bev=bev)
