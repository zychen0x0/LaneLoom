"""Resolve LaneLoom's compact temporal-sampling configuration."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class TemporalSamplingSpec:
    distance_targets_m: tuple[float, ...]
    fallback_frame_offsets: tuple[int, ...]

    @property
    def history_num_frames(self):
        return len(self.distance_targets_m)

    @property
    def queue_length(self):
        return self.history_num_frames + 1

    @property
    def frame_offsets(self):
        return tuple(-offset for offset in self.fallback_frame_offsets) + (0,)

    @property
    def temporal_gt_augmentation(self):
        return bool(self.distance_targets_m)


def resolve_temporal_sampling(value):
    """Derive all temporal settings from distances or ``False``.

    Each distance must be an integer number of metres because the same number
    is used as its fallback frame lookback (10 m -> t-10).
    """
    if value is False:
        return TemporalSamplingSpec((), ())
    if value is True or not isinstance(value, tuple):
        raise TypeError(
            'temporal_sampling_distance must be False or a tuple of '
            'positive integer distances in far-to-near order.')
    if not value:
        raise ValueError(
            'temporal_sampling_distance cannot be empty; use False for '
            'single-frame sampling.')

    targets = []
    for distance in value:
        if (isinstance(distance, bool) or not isinstance(distance, Real)
                or not math.isfinite(float(distance))):
            raise TypeError('temporal sampling distances must be finite numbers.')
        distance = float(distance)
        if distance <= 0 or not distance.is_integer():
            raise ValueError(
                'temporal sampling distances must be positive integer values '
                'so fallback frame offsets are unambiguous.')
        targets.append(distance)

    if any(older <= newer for older, newer in zip(targets, targets[1:])):
        raise ValueError(
            'temporal sampling distances must be unique and ordered from '
            'far to near, for example (10.0, 5.0).')

    target_tuple = tuple(targets)
    return TemporalSamplingSpec(
        distance_targets_m=target_tuple,
        fallback_frame_offsets=tuple(
            int(distance) for distance in target_tuple),
    )
