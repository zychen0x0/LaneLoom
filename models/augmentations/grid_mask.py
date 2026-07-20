"""A lightweight in-project GridMask implementation."""
import torch
import torch.nn as nn


class GridMask(nn.Module):
    """Randomly mask horizontal and vertical image stripes during training."""

    def __init__(self, ratio=0.5, prob=0.7):
        super().__init__()
        if not 0.0 <= ratio < 1.0:
            raise ValueError('ratio must lie in [0, 1).')
        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob must lie in [0, 1].')
        self.ratio = float(ratio)
        self.prob = float(prob)

    def forward(self, image):
        if not self.training or self.prob == 0.0 or torch.rand(()) >= self.prob:
            return image
        if image.ndim != 4:
            raise ValueError('GridMask expects image tensors shaped [B, C, H, W].')

        _, _, height, width = image.shape
        max_period = max(3, min(height, width) // 2)
        period = int(torch.randint(2, max_period, ()).item())
        stripe = max(1, int(period * self.ratio))
        shift_y = int(torch.randint(0, period, ()).item())
        shift_x = int(torch.randint(0, period, ()).item())
        rows = (torch.arange(height, device=image.device) + shift_y) % period
        cols = (torch.arange(width, device=image.device) + shift_x) % period
        keep = (rows[:, None] >= stripe) & (cols[None, :] >= stripe)
        return image * keep.to(dtype=image.dtype)[None, None]
