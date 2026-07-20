import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None

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
