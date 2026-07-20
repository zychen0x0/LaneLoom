# Copyright (c) 2022 The OpenLane Dataset Authors.
# Licensed under the Apache License, Version 2.0.
# Modified for LaneLoom's in-process MMEngine integration.
"""Geometry helpers from the official OpenLane 3D evaluator."""

import numpy as np
from scipy.interpolate import interp1d


def prune_3d_lane_by_visibility(lane_3d, visibility):
    return lane_3d[np.asarray(visibility) > 0]


def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    lane_3d = lane_3d[
        np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200)]
    return lane_3d[
        np.logical_and(lane_3d[:, 0] > x_min, lane_3d[:, 0] < x_max)]


def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """Interpolate a lane at OpenLane's fixed longitudinal samples."""
    if input_lane.shape[0] < 2:
        raise ValueError('A lane needs at least two points for resampling.')
    y_min = np.min(input_lane[:, 1]) - 5
    y_max = np.max(input_lane[:, 1]) + 5
    if input_lane.shape[1] < 3:
        input_lane = np.concatenate(
            [input_lane, np.zeros((input_lane.shape[0], 1), dtype=np.float32)],
            axis=1)
    x_values = interp1d(
        input_lane[:, 1], input_lane[:, 0],
        fill_value='extrapolate')(y_steps)
    z_values = interp1d(
        input_lane[:, 1], input_lane[:, 2],
        fill_value='extrapolate')(y_steps)
    if not out_vis:
        return x_values, z_values
    visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
    return x_values, z_values, visibility.astype(np.float32) + 1e-9
