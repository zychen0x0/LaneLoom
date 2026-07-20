"""Global-pose BEV shift and temporal attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from .bev_grid import bev_cell_centers


@MODELS.register_module()
class LaneLoomTemporalSelfAttention(nn.Module):
    """Shift global-pose-aligned history BEVs then attend over history time."""

    def __init__(self, embed_dims, bev_h, bev_w, pc_range, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.bev_h, self.bev_w = bev_h, bev_w
        self.pc_range = pc_range
        self.query_proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=False)
        self.key_proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=False)
        self.value_proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(embed_dims * 2, embed_dims, 1),
            nn.BatchNorm2d(embed_dims), nn.ReLU(inplace=True))

    @staticmethod
    def _meta(metas, index):
        if metas is None:
            return {}
        meta = metas[index] if isinstance(metas, (list, tuple)) else metas
        return meta.get('current', meta) if isinstance(meta, dict) else {}

    def _warp_to_current(self, bev, metas):
        """Use the history-ego -> current-ego transform derived via global pose."""
        batch, _, height, width = bev.shape
        y, x = bev_cell_centers(
            self.pc_range, height, width, bev.device, bev.dtype)
        current_grid = torch.stack((x, y, torch.zeros_like(x), torch.ones_like(x)), dim=-1)
        current_grid = current_grid.reshape(-1, 4).transpose(0, 1)
        grids = []
        for batch_idx in range(batch):
            transform = self._meta(metas, batch_idx).get('lidaradj2lidarcurr', torch.eye(4))
            transform = torch.as_tensor(transform, device=bev.device, dtype=bev.dtype)
            source = torch.linalg.solve(transform, current_grid)
            grid_x = ((source[0] - self.pc_range[0]) /
                      (self.pc_range[3] - self.pc_range[0]) * 2 - 1)
            grid_y = ((source[1] - self.pc_range[1]) /
                      (self.pc_range[4] - self.pc_range[1]) * 2 - 1)
            grids.append(torch.stack((grid_x, grid_y), dim=-1).reshape(height, width, 2))
        # Center coordinates map to -1 + 1/W ... 1 - 1/W when
        # ``align_corners=False``; identity pose therefore preserves cells.
        return F.grid_sample(bev, torch.stack(grids), align_corners=False)

    def forward(self, current_bev, prev_bev):
        if not prev_bev:
            return current_bev
        aligned = []
        for item in prev_bev:
            if isinstance(item, dict) and item.get('bev') is not None:
                aligned.append(self._warp_to_current(item['bev'], item.get('img_metas')))
        if not aligned:
            return current_bev
        history = torch.stack(aligned, dim=1)
        query = self.query_proj(current_bev).unsqueeze(1)
        keys = self.key_proj(history.flatten(0, 1)).reshape_as(history)
        values = self.value_proj(history.flatten(0, 1)).reshape_as(history)
        weights = ((query * keys).sum(2) / math.sqrt(self.embed_dims)).softmax(dim=1).unsqueeze(2)
        attended_history = (weights * values).sum(1)
        return current_bev + self.output(torch.cat((current_bev, attended_history), dim=1))
