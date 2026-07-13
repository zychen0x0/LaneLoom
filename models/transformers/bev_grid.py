"""Metric coordinates for the centers of BEV cells."""
import torch


def bev_cell_centers(pc_range, bev_h, bev_w, device, dtype):
    """Return y/x center-coordinate maps for a ``bev_h × bev_w`` BEV grid."""
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_step = (x_max - x_min) / bev_w
    y_step = (y_max - y_min) / bev_h
    x = x_min + (torch.arange(bev_w, device=device, dtype=dtype) + 0.5) * x_step
    y = y_min + (torch.arange(bev_h, device=device, dtype=dtype) + 0.5) * y_step
    return torch.meshgrid(y, x, indexing='ij')
