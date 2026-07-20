import numpy as np
import torch
import torch.nn as nn

try:
    import albumentations as A
except ImportError:
    A = None


class GridMask(nn.Module):
    """Randomly mask image stripes during training."""

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
            raise ValueError(
                'GridMask expects image tensors shaped [B, C, H, W].')

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


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
            if results is None:
                return None
        return results


class Albu:
    def __init__(self, transforms):
        if A is None:
            raise ImportError('albumentations is required by Albu.')
        self.transform = A.Compose([
            self._build_transform(transform_cfg)
            for transform_cfg in transforms
        ])

    def _build_transform(self, cfg):
        cfg = dict(cfg)
        transform_type = cfg.pop('type')
        nested = cfg.pop('transforms', None)
        cfg = {key: value for key, value in cfg.items() if value is not None}
        if transform_type == 'OneOf':
            return A.OneOf([self._build_transform(item) for item in nested], **cfg)
        return getattr(A, transform_type)(**cfg)

    def __call__(self, results):
        results['img'] = self.transform(image=results['img'])['image']
        return results


class PointsRangeFilter:
    def __init__(self, point_cloud_range):
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def __call__(self, results):
        points = results['points']
        pc_range = self.point_cloud_range
        mask = (
            (points[:, 0] >= pc_range[0])
            & (points[:, 1] >= pc_range[1])
            & (points[:, 2] >= pc_range[2])
            & (points[:, 0] <= pc_range[3])
            & (points[:, 1] <= pc_range[4])
            & (points[:, 2] <= pc_range[5])
        )
        results['points'] = points[mask]
        return results


class PointShuffle:
    def __call__(self, results):
        points = results['points']
        if points.shape[0] > 0:
            np.random.shuffle(points)
        results['points'] = points
        return results
