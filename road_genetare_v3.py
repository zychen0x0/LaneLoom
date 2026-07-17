#!/usr/bin/env python3
"""
Segment road-surface points with multiple road-arm height profiles.

Compared with road_genetare_v2.py, this version keeps the same conservative
position_range lateral extension, but avoids flattening steep uphill/downhill
extensions at the position_range z limits.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_fill_holes, gaussian_filter, gaussian_filter1d
from scipy.ndimage import label as label_connected_components
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)

from data_visualization import (
    find_frame_file,
    find_segment_name,
    find_split_name,
    guess_lidar_root,
    project_vehicle_points_to_image,
    repo_root,
    resolve_label_segment_dir,
    rotate_xy_ccw_90,
    save_lidar_fov_coverage_image,
    write_ply_xyzrgb,
)
from road_genetare import (
    DEFAULT_POSITION_RANGE,
    collect_images,
    estimate_sparse_ground_grid,
    fill_missing_with_nearest,
    load_single_frame_points,
    parse_floats,
    smooth_ground_grid,
)
from road_genetare_v1 import (
    ROAD_RGB,
    ROADSIDE_RGB,
    colorize_segmentation,
    keep_local_lowest_layer,
    plot_3d,
    plot_bev,
    plot_residual_hist,
)


EXTENDED_ROAD_RGB = np.array([60, 120, 245], dtype=np.uint8)


def estimate_ground_surface(points, position_range, grid_step, lowest_k, sigma_x, sigma_y):
    x_centers, y_centers, sparse_grid, in_range_points, occupied_cells = (
        estimate_sparse_ground_grid(
            points,
            position_range,
            grid_step,
            lowest_k,
        )
    )
    filled_grid = fill_missing_with_nearest(sparse_grid)
    smoothed_grid = smooth_ground_grid(filled_grid, sigma_x, sigma_y)
    interpolator = RegularGridInterpolator(
        (y_centers, x_centers),
        smoothed_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolator, smoothed_grid, in_range_points, occupied_cells


def rotate_xy_cw_90(xyz):
    rotated = xyz.copy()
    rotated[:, 0] = xyz[:, 1]
    rotated[:, 1] = -xyz[:, 0]
    return rotated


def estimate_sparse_low_road_grid(
    points,
    road_mask,
    position_range,
    grid_step,
    min_cell_points,
    z_percentile,
):
    xmin, ymin, zmin, xmax, ymax, zmax = position_range
    x_centers = np.arange(xmin + grid_step * 0.5, xmax, grid_step, dtype=np.float32)
    y_centers = np.arange(ymin + grid_step * 0.5, ymax, grid_step, dtype=np.float32)
    nx, ny = len(x_centers), len(y_centers)
    xyz = points[:, :3]
    idx = np.flatnonzero(
        road_mask
        & (xyz[:, 0] >= xmin)
        & (xyz[:, 0] < xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] < ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    if idx.size == 0:
        return x_centers, y_centers, grid, 0, 0

    ix = np.floor((xyz[idx, 0] - xmin) / grid_step).astype(np.int32)
    iy = np.floor((xyz[idx, 1] - ymin) / grid_step).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    idx, ix, iy = idx[valid], ix[valid], iy[valid]
    if idx.size == 0:
        return x_centers, y_centers, grid, 0, 0

    keys = iy * nx + ix
    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_z = xyz[idx[order], 2]
    unique_keys, starts = np.unique(sorted_keys, return_index=True)
    ends = np.r_[starts[1:], sorted_keys.size]
    occupied = 0
    for key, start, end in zip(unique_keys, starts, ends):
        values = sorted_z[start:end]
        if values.size < min_cell_points:
            continue
        cell_y = int(key // nx)
        cell_x = int(key % nx)
        grid[cell_y, cell_x] = np.percentile(values, z_percentile).astype(np.float32)
        occupied += 1
    return x_centers, y_centers, grid, int(idx.size), occupied


def refine_road_height_consistency(
    points,
    road_mask,
    position_range,
    grid_step,
    min_cell_points,
    z_percentile,
    below_margin,
    above_margin,
    smooth_sigma,
):
    if grid_step <= 0 or not np.any(road_mask):
        return road_mask, {"enabled": False, "reason": "empty_or_disabled"}

    x_centers, y_centers, sparse_grid, candidate_points, occupied_cells = (
        estimate_sparse_low_road_grid(
            points,
            road_mask,
            position_range,
            grid_step,
            min_cell_points,
            z_percentile,
        )
    )
    if occupied_cells < 4:
        return road_mask, {
            "enabled": False,
            "reason": "too_few_reference_cells",
            "candidate_points": int(candidate_points),
            "reference_cells": int(occupied_cells),
        }

    filled_grid = fill_missing_with_nearest(sparse_grid)
    sigma_cells = max(0.0, smooth_sigma / grid_step)
    smoothed_grid = smooth_ground_grid(filled_grid, sigma_cells, sigma_cells)
    interpolator = RegularGridInterpolator(
        (y_centers, x_centers),
        smoothed_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
    road_idx = np.flatnonzero(road_mask)
    xyz = points[road_idx, :3]
    expected_z = interpolator(np.stack([xyz[:, 1], xyz[:, 0]], axis=1))
    residual = xyz[:, 2] - expected_z
    keep_road = (
        np.isfinite(residual)
        & (residual >= -below_margin)
        & (residual <= above_margin)
    )
    refined = road_mask.copy()
    refined[road_idx] = keep_road
    details = {
        "enabled": True,
        "candidate_points": int(candidate_points),
        "reference_cells": int(occupied_cells),
        "before_points": int(road_mask.sum()),
        "after_points": int(refined.sum()),
        "removed_points": int(road_mask.sum() - refined.sum()),
        "grid_step": float(grid_step),
        "min_cell_points": int(min_cell_points),
        "z_percentile": float(z_percentile),
        "below_margin": float(below_margin),
        "above_margin": float(above_margin),
        "smooth_sigma": float(smooth_sigma),
    }
    return refined, details


def extrapolate_profile(query_y, source_y, source_values, edge_bins):
    values = np.interp(query_y, source_y, source_values).astype(np.float32)
    edge_count = min(max(2, int(edge_bins)), source_y.size)
    if edge_count >= 2:
        left_y = source_y[:edge_count]
        right_y = source_y[-edge_count:]
        if float(left_y[-1] - left_y[0]) > 1e-3:
            left_coef = np.polyfit(left_y, source_values[:edge_count], 1)
            left_mask = query_y < source_y[0]
            values[left_mask] = np.polyval(left_coef, query_y[left_mask]).astype(np.float32)
        if float(right_y[-1] - right_y[0]) > 1e-3:
            right_coef = np.polyfit(right_y, source_values[-edge_count:], 1)
            right_mask = query_y > source_y[-1]
            values[right_mask] = np.polyval(right_coef, query_y[right_mask]).astype(np.float32)
    return values


def build_centerline_extended_surface(
    road_points,
    position_range,
    step_xy,
    bin_size,
    min_bin_points,
    width_percentile,
    width_padding,
    min_width,
    max_width,
    smooth_sigma,
    extrapolate_bins,
):
    xmin, ymin, zmin, xmax, ymax, zmax = position_range
    step_x, step_y = step_xy
    if step_x <= 0 or step_y <= 0 or bin_size <= 0:
        raise ValueError("extension step and bin size must be positive.")
    if road_points.size == 0:
        raise ValueError("No extracted road points to extend.")

    xyz = road_points[:, :3]
    in_range = (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] < xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] < ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )
    xyz = xyz[in_range]
    if xyz.shape[0] < min_bin_points:
        raise ValueError("Too few extracted road points to extend.")

    y_bins = np.arange(ymin, ymax + bin_size, bin_size, dtype=np.float32)
    y_centers = y_bins[:-1] + bin_size * 0.5
    bin_idx = np.floor((xyz[:, 1] - ymin) / bin_size).astype(np.int32)
    valid_idx = (bin_idx >= 0) & (bin_idx < y_centers.size)
    bin_idx = bin_idx[valid_idx]
    xyz = xyz[valid_idx]

    profile_y = []
    center_x = []
    half_width = []
    profile_z = []
    counts = []
    lo_pct = width_percentile
    hi_pct = 100.0 - width_percentile
    for idx in np.unique(bin_idx):
        values = xyz[bin_idx == idx]
        if values.shape[0] < min_bin_points:
            continue
        left = np.percentile(values[:, 0], lo_pct)
        right = np.percentile(values[:, 0], hi_pct)
        width = max(0.0, float(right - left)) + width_padding * 2.0
        if min_width > 0:
            width = max(width, min_width)
        if max_width > 0:
            width = min(width, max_width)
        profile_y.append(float(y_centers[idx]))
        center_x.append(float(np.median(values[:, 0])))
        half_width.append(width * 0.5)
        profile_z.append(float(np.median(values[:, 2])))
        counts.append(int(values.shape[0]))

    if len(profile_y) < 2:
        raise ValueError("Too few longitudinal road bins to extend.")

    profile_y = np.asarray(profile_y, dtype=np.float32)
    center_x = np.asarray(center_x, dtype=np.float32)
    half_width = np.asarray(half_width, dtype=np.float32)
    profile_z = np.asarray(profile_z, dtype=np.float32)

    out_y = np.arange(ymin, ymax + step_y * 0.5, step_y, dtype=np.float32)
    out_center = extrapolate_profile(out_y, profile_y, center_x, extrapolate_bins)
    out_z = extrapolate_profile(out_y, profile_y, profile_z, extrapolate_bins)
    out_half_width = np.interp(out_y, profile_y, half_width).astype(np.float32)
    sigma_rows = max(0.0, smooth_sigma / step_y)
    if sigma_rows > 0:
        out_center = gaussian_filter1d(out_center, sigma=sigma_rows, mode="nearest")
        out_z = gaussian_filter1d(out_z, sigma=sigma_rows, mode="nearest")
        out_half_width = gaussian_filter1d(out_half_width, sigma=sigma_rows, mode="nearest")

    rows = []
    for y_value, cx, z_value, hw in zip(out_y, out_center, out_z, out_half_width):
        if not np.isfinite(cx) or not np.isfinite(z_value) or not np.isfinite(hw):
            continue
        if z_value < zmin or z_value > zmax or hw <= 0:
            continue
        lateral = np.arange(-hw, hw + step_x * 0.5, step_x, dtype=np.float32)
        x = cx + lateral
        valid = (x >= xmin) & (x <= xmax)
        if not np.any(valid):
            continue
        y = np.full(int(valid.sum()), y_value, dtype=np.float32)
        z = np.full(int(valid.sum()), z_value, dtype=np.float32)
        rows.append(np.stack([x[valid], y, z], axis=1))

    if not rows:
        raise ValueError("No extended road-band points were generated.")

    extended = np.concatenate(rows, axis=0).astype(np.float32)
    details = {
        "mode": "centerline_road_band",
        "source_points": int(road_points.shape[0]),
        "source_points_in_range": int(xyz.shape[0]),
        "profile_bins": int(profile_y.size),
        "profile_y_min": float(profile_y.min()),
        "profile_y_max": float(profile_y.max()),
        "generated_points": int(extended.shape[0]),
        "bin_size": float(bin_size),
        "min_bin_points": int(min_bin_points),
        "width_percentile": float(width_percentile),
        "width_padding": float(width_padding),
        "min_width": float(min_width),
        "max_width": float(max_width),
        "smooth_sigma": float(smooth_sigma),
        "extrapolate_bins": int(extrapolate_bins),
        "bin_point_count_min": int(np.min(counts)),
        "bin_point_count_max": int(np.max(counts)),
    }
    return extended, details


def build_position_range_y_profile_surface(
    road_points,
    position_range,
    step_xy,
    bin_size,
    min_bin_points,
    smooth_sigma,
    extrapolate_bins,
    z_clip_margin,
):
    xmin, ymin, zmin, xmax, ymax, zmax = position_range
    step_x, step_y = step_xy
    if step_x <= 0 or step_y <= 0 or bin_size <= 0:
        raise ValueError("extension step and bin size must be positive.")
    if road_points.size == 0:
        raise ValueError("No extracted road points to extend.")

    xyz = road_points[:, :3]
    in_range = (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] < xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] < ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )
    xyz = xyz[in_range]
    if xyz.shape[0] < min_bin_points:
        raise ValueError("Too few extracted road points to estimate a y-height profile.")

    y_bins = np.arange(ymin, ymax + bin_size, bin_size, dtype=np.float32)
    y_centers = y_bins[:-1] + bin_size * 0.5
    bin_idx = np.floor((xyz[:, 1] - ymin) / bin_size).astype(np.int32)
    valid_idx = (bin_idx >= 0) & (bin_idx < y_centers.size)
    bin_idx = bin_idx[valid_idx]
    xyz = xyz[valid_idx]

    profile_y = []
    profile_z = []
    counts = []
    for idx in np.unique(bin_idx):
        values = xyz[bin_idx == idx]
        if values.shape[0] < min_bin_points:
            continue
        profile_y.append(float(y_centers[idx]))
        profile_z.append(float(np.median(values[:, 2])))
        counts.append(int(values.shape[0]))

    if len(profile_y) < 2:
        raise ValueError("Too few longitudinal road bins to extend.")

    profile_y = np.asarray(profile_y, dtype=np.float32)
    profile_z = np.asarray(profile_z, dtype=np.float32)
    out_y = np.arange(ymin, ymax + step_y * 0.5, step_y, dtype=np.float32)
    out_z = extrapolate_profile(out_y, profile_y, profile_z, extrapolate_bins)
    sigma_rows = max(0.0, smooth_sigma / step_y)
    if sigma_rows > 0:
        out_z = gaussian_filter1d(out_z, sigma=sigma_rows, mode="nearest")
    extension_zmin = zmin - max(0.0, float(z_clip_margin))
    extension_zmax = zmax + max(0.0, float(z_clip_margin))
    out_z = np.clip(out_z, extension_zmin, extension_zmax).astype(np.float32)

    out_x = np.arange(xmin, xmax + step_x * 0.5, step_x, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(out_x, out_y)
    z_grid = np.repeat(out_z.reshape(-1, 1), out_x.size, axis=1)
    extended = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    ).astype(np.float32)

    details = {
        "mode": "position_range_y_profile",
        "source_points": int(road_points.shape[0]),
        "source_points_in_range": int(xyz.shape[0]),
        "profile_bins": int(profile_y.size),
        "profile_y_min": float(profile_y.min()),
        "profile_y_max": float(profile_y.max()),
        "generated_points": int(extended.shape[0]),
        "x_min": float(xmin),
        "x_max": float(xmax),
        "y_min": float(ymin),
        "y_max": float(ymax),
        "bin_size": float(bin_size),
        "min_bin_points": int(min_bin_points),
        "smooth_sigma": float(smooth_sigma),
        "extrapolate_bins": int(extrapolate_bins),
        "z_clip_margin": float(z_clip_margin),
        "z_clip_min": float(extension_zmin),
        "z_clip_max": float(extension_zmax),
        "generated_z_min": float(out_z.min()),
        "generated_z_max": float(out_z.max()),
        "bin_point_count_min": int(np.min(counts)),
        "bin_point_count_max": int(np.max(counts)),
    }
    return extended, details


def projection_limit_to_range(root_xy, direction, position_range):
    xmin, ymin, _, xmax, ymax, _ = position_range
    corners = np.array(
        [
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymin],
            [xmax, ymax],
        ],
        dtype=np.float32,
    )
    s = (corners - root_xy.reshape(1, 2)) @ direction
    return max(0.0, float(np.max(s)))


def build_arm_profile_extended_surface(
    road_points,
    root_xy,
    profiles,
    position_range,
    step_xy,
    bin_size,
    min_bin_points,
    width_percentile,
    width_padding,
    min_width,
    max_width,
    smooth_sigma,
    extrapolate_bins,
    assign_half_width,
    min_forward_component,
):
    if not profiles:
        raise ValueError("No road-arm profiles to extend.")

    xmin, ymin, zmin, xmax, ymax, zmax = position_range
    step_x, step_s = step_xy
    xyz = road_points[:, :3]
    vec = xyz[:, :2] - root_xy.reshape(1, 2)
    point_angles = (np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0

    blocks = []
    arm_details = []
    lo_pct = width_percentile
    hi_pct = 100.0 - width_percentile
    for profile in profiles:
        direction = np.asarray(profile["direction"], dtype=np.float32)
        if float(direction[1]) < min_forward_component:
            continue
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        s_all = vec @ direction
        assigned = (
            (s_all >= 0.0)
            & (circular_angle_distance(point_angles, float(profile["angle_deg"])) <= assign_half_width)
        )
        arm_xyz = xyz[assigned]
        arm_s = s_all[assigned]
        if arm_xyz.shape[0] < min_bin_points:
            continue
        arm_t = (arm_xyz[:, :2] - root_xy.reshape(1, 2)) @ normal
        arm_z = arm_xyz[:, 2]
        s_limit = projection_limit_to_range(root_xy, direction, position_range)
        if s_limit <= 0:
            continue

        s_bins = np.arange(0.0, max(float(np.max(arm_s)), bin_size) + bin_size, bin_size, dtype=np.float32)
        s_centers = s_bins[:-1] + bin_size * 0.5
        bin_idx = np.floor(arm_s / bin_size).astype(np.int32)
        valid_idx = (bin_idx >= 0) & (bin_idx < s_centers.size)
        bin_idx = bin_idx[valid_idx]
        arm_t_valid = arm_t[valid_idx]
        arm_z_valid = arm_z[valid_idx]

        profile_s = []
        center_t = []
        half_width = []
        profile_z = []
        counts = []
        for idx in np.unique(bin_idx):
            t_values = arm_t_valid[bin_idx == idx]
            z_values = arm_z_valid[bin_idx == idx]
            if t_values.size < min_bin_points:
                continue
            left = np.percentile(t_values, lo_pct)
            right = np.percentile(t_values, hi_pct)
            width = max(0.0, float(right - left)) + width_padding * 2.0
            if min_width > 0:
                width = max(width, min_width)
            if max_width > 0:
                width = min(width, max_width)
            profile_s.append(float(s_centers[idx]))
            center_t.append(float(np.median(t_values)))
            half_width.append(width * 0.5)
            profile_z.append(float(np.median(z_values)))
            counts.append(int(t_values.size))

        if len(profile_s) < 2:
            continue

        profile_s = np.asarray(profile_s, dtype=np.float32)
        center_t = np.asarray(center_t, dtype=np.float32)
        half_width = np.asarray(half_width, dtype=np.float32)
        profile_z = np.asarray(profile_z, dtype=np.float32)
        out_s = np.arange(0.0, s_limit + step_s * 0.5, step_s, dtype=np.float32)
        out_center_t = extrapolate_profile(out_s, profile_s, center_t, extrapolate_bins)
        out_z = extrapolate_profile(out_s, profile_s, profile_z, extrapolate_bins)
        out_half_width = np.interp(out_s, profile_s, half_width).astype(np.float32)
        sigma_rows = max(0.0, smooth_sigma / step_s)
        if sigma_rows > 0:
            out_center_t = gaussian_filter1d(out_center_t, sigma=sigma_rows, mode="nearest")
            out_z = gaussian_filter1d(out_z, sigma=sigma_rows, mode="nearest")
            out_half_width = gaussian_filter1d(out_half_width, sigma=sigma_rows, mode="nearest")

        rows = []
        for s_value, ct, z_value, hw in zip(out_s, out_center_t, out_z, out_half_width):
            if not np.isfinite(ct) or not np.isfinite(z_value) or not np.isfinite(hw):
                continue
            if z_value < zmin or z_value > zmax or hw <= 0:
                continue
            lateral = np.arange(ct - hw, ct + hw + step_x * 0.5, step_x, dtype=np.float32)
            xy = (
                root_xy.reshape(1, 2)
                + s_value * direction.reshape(1, 2)
                + lateral.reshape(-1, 1) * normal.reshape(1, 2)
            )
            valid = (
                (xy[:, 0] >= xmin)
                & (xy[:, 0] <= xmax)
                & (xy[:, 1] >= ymin)
                & (xy[:, 1] <= ymax)
            )
            if not np.any(valid):
                continue
            z = np.full(int(valid.sum()), z_value, dtype=np.float32)
            rows.append(np.column_stack([xy[valid], z]).astype(np.float32))

        if not rows:
            continue
        arm_points = np.concatenate(rows, axis=0).astype(np.float32)
        blocks.append(arm_points)
        arm_details.append(
            {
                "id": int(profile["id"]),
                "angle_deg": float(profile["angle_deg"]),
                "forward_component": float(direction[1]),
                "source_points": int(arm_xyz.shape[0]),
                "profile_bins": int(profile_s.size),
                "s_limit": float(s_limit),
                "generated_points": int(arm_points.shape[0]),
                "bin_point_count_min": int(np.min(counts)),
                "bin_point_count_max": int(np.max(counts)),
            }
        )

    if not blocks:
        raise ValueError("No forward road-arm extension points were generated.")
    extended = np.concatenate(blocks, axis=0)
    keys = np.round(extended[:, :2] / max(step_x * 0.5, 1e-3)).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    extended = extended[np.sort(keep)]
    details = {
        "mode": "forward_angle_arm_road_band",
        "source_points": int(road_points.shape[0]),
        "generated_points": int(extended.shape[0]),
        "min_forward_component": float(min_forward_component),
        "arms": arm_details,
    }
    return extended.astype(np.float32), details


def repeat_rgb(rgb, count):
    return np.repeat(rgb.reshape(1, 3), int(count), axis=0)


def angle_distance_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def estimate_arm_root(points, candidate_mask, seed_x_abs, seed_y_min, seed_y_max):
    xyz = points[:, :3]
    seed = (
        candidate_mask
        & (np.abs(xyz[:, 0]) <= seed_x_abs)
        & (xyz[:, 1] >= seed_y_min)
        & (xyz[:, 1] <= seed_y_max)
    )
    if int(seed.sum()) >= 10:
        return np.array(
            [np.median(xyz[seed, 0]), np.percentile(xyz[seed, 1], 20.0)],
            dtype=np.float32,
        )

    fallback = (
        candidate_mask
        & (np.abs(xyz[:, 0]) <= max(seed_x_abs, 8.0))
        & (xyz[:, 1] >= 0.0)
        & (xyz[:, 1] <= 8.0)
    )
    if int(fallback.sum()) >= 10:
        return np.array(
            [np.median(xyz[fallback, 0]), np.median(xyz[fallback, 1])],
            dtype=np.float32,
        )
    return np.array([0.0, max(0.0, seed_y_min)], dtype=np.float32)


def build_candidate_bev_mask(
    points,
    candidate_mask,
    position_range,
    cell_size,
    min_cell_points,
    density_sigma,
    density_threshold,
    close_radius,
    open_radius,
    min_component_area,
    seed_x_abs,
    seed_y_min,
    seed_y_max,
):
    xmin, ymin, _, xmax, ymax, _ = position_range
    nx = int(np.ceil((xmax - xmin) / cell_size))
    ny = int(np.ceil((ymax - ymin) / cell_size))
    xyz = points[:, :3]

    idx = np.flatnonzero(candidate_mask)
    counts = np.zeros((ny, nx), dtype=np.int32)
    if idx.size == 0:
        return counts, counts.astype(np.float32), np.zeros_like(counts, dtype=bool)

    ix = np.floor((xyz[idx, 0] - xmin) / cell_size).astype(np.int32)
    iy = np.floor((xyz[idx, 1] - ymin) / cell_size).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(counts, (iy[valid], ix[valid]), 1)

    density = counts.astype(np.float32)
    if density_sigma > 0:
        density = gaussian_filter(
            density,
            sigma=max(0.1, density_sigma / cell_size),
            mode="nearest",
        ).astype(np.float32)

    mask = (counts >= max(1, int(min_cell_points))) | (density >= density_threshold)
    if close_radius > 0:
        mask = binary_closing(mask, footprint=disk(max(1, int(round(close_radius / cell_size)))))
    if open_radius > 0:
        mask = binary_opening(mask, footprint=disk(max(1, int(round(open_radius / cell_size)))))
    mask = binary_fill_holes(mask)

    min_size = max(1, int(round(min_component_area / (cell_size * cell_size))))
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)

    labels, num_labels = label_connected_components(mask)
    if num_labels <= 1:
        return counts, density, mask.astype(bool)

    yy, xx = np.mgrid[0:ny, 0:nx]
    x = xmin + (xx.astype(np.float32) + 0.5) * cell_size
    y = ymin + (yy.astype(np.float32) + 0.5) * cell_size
    seed_cells = (
        mask
        & (np.abs(x) <= seed_x_abs)
        & (y >= seed_y_min)
        & (y <= seed_y_max)
    )
    keep_labels = set(labels[seed_cells].tolist()) - {0}
    if not keep_labels:
        sizes = np.bincount(labels.ravel(), minlength=num_labels + 1)
        sizes[0] = 0
        keep_labels = {int(np.argmax(sizes))}

    mask = np.isin(labels, list(keep_labels))
    return counts, density, mask.astype(bool)


def circular_angle_distance(values, angle):
    return np.abs((values - angle + 180.0) % 360.0 - 180.0)


def detect_angle_arms(
    points,
    candidate_mask,
    root_xy,
    min_radius,
    bin_deg,
    smooth_sigma_deg,
    min_score_ratio,
    min_support_points,
    min_angle_separation,
    max_arms,
    sector_half_width,
):
    xyz = points[:, :3]
    candidate_xy = xyz[candidate_mask, :2]
    if candidate_xy.shape[0] == 0:
        return [], {"hist_angles": [], "hist_scores": []}

    vec = candidate_xy - root_xy.reshape(1, 2)
    radius = np.linalg.norm(vec, axis=1)
    valid = radius >= min_radius
    if int(valid.sum()) < min_support_points:
        return [], {"hist_angles": [], "hist_scores": []}

    angles = (np.rad2deg(np.arctan2(vec[valid, 1], vec[valid, 0])) + 360.0) % 360.0
    radius_valid = radius[valid]
    weights = np.minimum(radius_valid, np.percentile(radius_valid, 95.0))
    bins = np.arange(0.0, 360.0 + bin_deg, bin_deg, dtype=np.float32)
    hist, edges = np.histogram(angles, bins=bins, weights=weights)
    smooth = hist.astype(np.float32)
    if smooth_sigma_deg > 0:
        smooth = gaussian_filter1d(
            smooth,
            sigma=max(0.1, smooth_sigma_deg / bin_deg),
            mode="wrap",
        ).astype(np.float32)

    centers = (edges[:-1] + edges[1:]) * 0.5
    if smooth.size == 0 or float(smooth.max()) <= 0:
        return [], {"hist_angles": centers.tolist(), "hist_scores": smooth.tolist()}

    local_max = (smooth >= np.roll(smooth, 1)) & (smooth >= np.roll(smooth, -1))
    peak_idx = np.flatnonzero(local_max & (smooth >= float(smooth.max()) * min_score_ratio))
    order = peak_idx[np.argsort(smooth[peak_idx])[::-1]]

    arms = []
    all_vec = xyz[:, :2] - root_xy.reshape(1, 2)
    all_radius = np.linalg.norm(all_vec, axis=1)
    all_angles = (np.rad2deg(np.arctan2(all_vec[:, 1], all_vec[:, 0])) + 360.0) % 360.0
    for idx in order:
        angle = float(centers[idx] % 360.0)
        if any(angle_distance_deg(angle, arm["angle_deg"]) < min_angle_separation for arm in arms):
            continue
        support = (
            candidate_mask
            & (all_radius >= min_radius)
            & (circular_angle_distance(all_angles, angle) <= sector_half_width)
        )
        support_points = int(support.sum())
        if support_points < min_support_points:
            continue
        arms.append(
            {
                "id": len(arms),
                "angle_deg": angle,
                "score": float(smooth[idx]),
                "support_points": support_points,
            }
        )
        if len(arms) >= max_arms:
            break

    return arms, {"hist_angles": centers.tolist(), "hist_scores": smooth.tolist()}


def build_angle_arm_profiles(
    points,
    candidate_mask,
    root_xy,
    angle_arms,
    sector_half_width,
    bin_size,
    min_bin_points,
    smooth_sigma,
    min_profile_points,
):
    xyz = points[:, :3]
    vec = xyz[:, :2] - root_xy.reshape(1, 2)
    point_angles = (np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
    profiles = []

    for arm in angle_arms:
        angle_rad = np.deg2rad(arm["angle_deg"])
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)
        s_all = vec @ direction
        support = (
            candidate_mask
            & (s_all >= 0.0)
            & (circular_angle_distance(point_angles, arm["angle_deg"]) <= sector_half_width)
        )
        if int(support.sum()) < min_profile_points:
            continue

        s = s_all[support]
        z = xyz[support, 2]
        max_s = max(float(np.percentile(s, 98.0)), bin_size)
        centers = np.arange(bin_size * 0.5, max_s + bin_size * 0.5, bin_size, dtype=np.float32)
        profile_z = np.full(centers.shape, np.nan, dtype=np.float32)
        bin_idx = np.floor(s / bin_size).astype(np.int32)
        valid = (bin_idx >= 0) & (bin_idx < centers.size)
        for idx in np.unique(bin_idx[valid]):
            values = z[valid][bin_idx[valid] == idx]
            if values.size >= min_bin_points:
                profile_z[idx] = np.median(values).astype(np.float32)

        valid_bins = np.isfinite(profile_z)
        if int(valid_bins.sum()) < 2:
            continue

        filled = np.interp(
            centers,
            centers[valid_bins],
            profile_z[valid_bins],
            left=float(profile_z[valid_bins][0]),
            right=float(profile_z[valid_bins][-1]),
        ).astype(np.float32)
        if smooth_sigma > 0:
            filled = gaussian_filter1d(
                filled,
                sigma=max(0.1, smooth_sigma / bin_size),
                mode="nearest",
            ).astype(np.float32)

        profile = dict(arm)
        profile.update(
            {
                "direction": direction,
                "profile_s": centers,
                "profile_z": filled,
                "valid_profile_bins": int(valid_bins.sum()),
                "support_points": int(support.sum()),
                "max_s": float(max_s),
            }
        )
        profiles.append(profile)
    return profiles


def keep_by_angle_arm_profiles(
    points,
    candidate_mask,
    root_xy,
    profiles,
    assign_half_width,
    margin,
):
    xyz = points[:, :3]
    vec = xyz[:, :2] - root_xy.reshape(1, 2)
    point_angles = (np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
    keep = np.zeros(points.shape[0], dtype=bool)
    details = []

    for profile in profiles:
        direction = profile["direction"]
        s_all = vec @ direction
        near = (
            candidate_mask
            & (s_all >= 0.0)
            & (circular_angle_distance(point_angles, profile["angle_deg"]) <= assign_half_width)
        )
        if not np.any(near):
            details.append(
                {
                    "id": int(profile["id"]),
                    "angle_deg": float(profile["angle_deg"]),
                    "support_points": int(profile["support_points"]),
                    "kept_points": 0,
                }
            )
            continue

        expected_z = np.interp(
            s_all[near],
            profile["profile_s"],
            profile["profile_z"],
            left=float(profile["profile_z"][0]),
            right=float(profile["profile_z"][-1]),
        ).astype(np.float32)
        near_idx = np.flatnonzero(near)
        passed_idx = near_idx[np.abs(xyz[near, 2] - expected_z) <= margin]
        keep[passed_idx] = True
        details.append(
            {
                "id": int(profile["id"]),
                "angle_deg": float(profile["angle_deg"]),
                "score": float(profile["score"]),
                "support_points": int(profile["support_points"]),
                "valid_profile_bins": int(profile["valid_profile_bins"]),
                "kept_points": int(passed_idx.size),
                "max_s": float(profile["max_s"]),
            }
        )
    return keep, details


def refine_road_by_angle_profile_height(
    points,
    road_mask,
    root_xy,
    profiles,
    assign_half_width,
    below_margin,
    above_margin,
):
    if not profiles or not np.any(road_mask):
        return road_mask, {"enabled": False, "reason": "empty_profiles_or_mask"}

    xyz = points[:, :3]
    vec = xyz[:, :2] - root_xy.reshape(1, 2)
    point_angles = (np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
    assigned = np.zeros(points.shape[0], dtype=bool)
    passed = np.zeros(points.shape[0], dtype=bool)

    for profile in profiles:
        direction = profile["direction"]
        s_all = vec @ direction
        near = (
            road_mask
            & (s_all >= 0.0)
            & (circular_angle_distance(point_angles, profile["angle_deg"]) <= assign_half_width)
        )
        if not np.any(near):
            continue
        expected_z = np.interp(
            s_all[near],
            profile["profile_s"],
            profile["profile_z"],
            left=float(profile["profile_z"][0]),
            right=float(profile["profile_z"][-1]),
        ).astype(np.float32)
        near_idx = np.flatnonzero(near)
        residual = xyz[near, 2] - expected_z
        assigned[near_idx] = True
        passed[near_idx] |= (residual >= -below_margin) & (residual <= above_margin)

    refined = road_mask.copy()
    assigned_road = road_mask & assigned
    refined[assigned_road] = passed[assigned_road]
    details = {
        "enabled": True,
        "before_points": int(road_mask.sum()),
        "assigned_points": int(assigned_road.sum()),
        "unassigned_points": int((road_mask & ~assigned).sum()),
        "after_points": int(refined.sum()),
        "removed_points": int(road_mask.sum() - refined.sum()),
        "below_margin": float(below_margin),
        "above_margin": float(above_margin),
        "assign_half_width": float(assign_half_width),
    }
    return refined, details


def keep_connected_to_arm_area(
    points,
    road_mask,
    arm_mask,
    position_range,
    cell_size,
    min_cell_points,
):
    if cell_size <= 0 or not np.any(road_mask):
        return road_mask

    xmin, ymin, _, xmax, ymax, _ = position_range
    nx = int(np.ceil((xmax - xmin) / cell_size))
    ny = int(np.ceil((ymax - ymin) / cell_size))
    xyz = points[:, :3]
    road_idx = np.flatnonzero(road_mask)
    ix = np.floor((xyz[road_idx, 0] - xmin) / cell_size).astype(np.int32)
    iy = np.floor((xyz[road_idx, 1] - ymin) / cell_size).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.any(valid):
        return road_mask

    counts = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(counts, (iy[valid], ix[valid]), 1)
    occupied = counts >= max(1, int(min_cell_points))
    labels, num_labels = label_connected_components(occupied, structure=np.ones((3, 3), dtype=np.uint8))
    if num_labels == 0:
        return road_mask

    arm_labels = set(labels[arm_mask & (labels > 0)].tolist())
    if not arm_labels:
        weighted_counts = np.bincount(labels.ravel(), weights=counts.ravel(), minlength=num_labels + 1)
        weighted_counts[0] = 0
        arm_labels = {int(np.argmax(weighted_counts))}

    keep_road = valid & np.isin(labels[iy, ix], list(arm_labels))
    keep = road_mask.copy()
    keep[road_idx] = keep_road
    return keep


def angle_profiles_to_mask(profiles, root_xy, position_range, cell_size, half_width_deg):
    xmin, ymin, _, xmax, ymax, _ = position_range
    nx = int(np.ceil((xmax - xmin) / cell_size))
    ny = int(np.ceil((ymax - ymin) / cell_size))
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = xmin + (xx.astype(np.float32) + 0.5) * cell_size
    y = ymin + (yy.astype(np.float32) + 0.5) * cell_size
    vec_x = x - float(root_xy[0])
    vec_y = y - float(root_xy[1])
    cell_angles = (np.rad2deg(np.arctan2(vec_y, vec_x)) + 360.0) % 360.0
    mask = np.zeros((ny, nx), dtype=bool)
    for profile in profiles:
        s = vec_x * float(profile["direction"][0]) + vec_y * float(profile["direction"][1])
        mask |= (
            (s >= 0.0)
            & (s <= float(profile["max_s"]))
            & (circular_angle_distance(cell_angles, float(profile["angle_deg"])) <= half_width_deg)
        )
    return mask


def segment_points(
    points,
    ground_interp,
    position_range,
    residual_threshold,
    road_below_threshold,
    road_above_threshold,
    local_lowest_cell_size,
    local_lowest_margin,
    profile_bin_size,
    profile_seed_x_abs,
    profile_margin,
    profile_min_seed_points,
    arm_cell_size,
    arm_density_min_cell_points,
    arm_density_sigma,
    arm_density_threshold,
    arm_close_radius,
    arm_open_radius,
    arm_min_component_area,
    arm_min_angle_separation,
    arm_max_count,
    arm_profile_min_bin_points,
    arm_profile_smooth_sigma,
    arm_profile_min_points,
    angle_arm_min_radius,
    angle_arm_bin_deg,
    angle_arm_smooth_sigma_deg,
    angle_arm_min_score_ratio,
    angle_arm_min_support_points,
    angle_arm_sector_half_width,
    angle_arm_assign_half_width,
    near_field_radius,
    near_field_y_max,
    height_consistency_cell_size,
    height_consistency_min_cell_points,
    height_consistency_percentile,
    height_consistency_below_margin,
    height_consistency_above_margin,
    height_consistency_smooth_sigma,
    connected_cell_size,
    connected_min_cell_points,
):
    xmin, ymin, zmin, xmax, ymax, zmax = position_range
    xyz = points[:, :3]
    range_mask = (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] < xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] < ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )
    points_in_range = points[range_mask]
    query = np.stack([points_in_range[:, 1], points_in_range[:, 0]], axis=1)
    ground_z = ground_interp(query)
    valid = np.isfinite(ground_z)

    points_in_range = points_in_range[valid]
    residual = points_in_range[:, 2] - ground_z[valid]
    raw_residual_mask = np.abs(residual) <= residual_threshold
    strict_residual_mask = (
        raw_residual_mask
        & (residual >= -road_below_threshold)
        & (residual <= road_above_threshold)
    )

    root_xy = estimate_arm_root(
        points_in_range,
        strict_residual_mask,
        profile_seed_x_abs,
        0.0,
        8.0,
    )
    counts, density, bev_mask = build_candidate_bev_mask(
        points_in_range,
        strict_residual_mask,
        position_range,
        arm_cell_size,
        arm_density_min_cell_points,
        arm_density_sigma,
        arm_density_threshold,
        arm_close_radius,
        arm_open_radius,
        arm_min_component_area,
        profile_seed_x_abs,
        0.0,
        30.0,
    )
    angle_arms, angle_hist = detect_angle_arms(
        points_in_range,
        strict_residual_mask,
        root_xy,
        angle_arm_min_radius,
        angle_arm_bin_deg,
        angle_arm_smooth_sigma_deg,
        angle_arm_min_score_ratio,
        angle_arm_min_support_points,
        arm_min_angle_separation,
        arm_max_count,
        angle_arm_sector_half_width,
    )
    angle_profiles = build_angle_arm_profiles(
        points_in_range,
        strict_residual_mask,
        root_xy,
        angle_arms,
        angle_arm_sector_half_width,
        profile_bin_size,
        arm_profile_min_bin_points,
        arm_profile_smooth_sigma,
        arm_profile_min_points,
    )

    if angle_profiles:
        road_mask, profile_details = keep_by_angle_arm_profiles(
            points_in_range,
            strict_residual_mask,
            root_xy,
            angle_profiles,
            angle_arm_assign_half_width,
            profile_margin,
        )
        profile_mode = "angle_arm_profiles"
    else:
        road_mask = strict_residual_mask.copy()
        profile_details = []
        profile_mode = "strict_residual_fallback"

    xyz_in_range = points_in_range[:, :3]
    root_vec = xyz_in_range[:, :2] - root_xy.reshape(1, 2)
    near_field_spatial_mask = np.zeros_like(strict_residual_mask, dtype=bool)
    if near_field_radius > 0:
        near_field_spatial_mask |= np.linalg.norm(root_vec, axis=1) <= near_field_radius
    if near_field_y_max > 0:
        near_field_spatial_mask |= xyz_in_range[:, 1] <= near_field_y_max
    near_field_mask = near_field_spatial_mask & raw_residual_mask
    near_field_strict_mask = near_field_spatial_mask & strict_residual_mask
    road_mask |= near_field_mask

    after_profile = int(road_mask.sum())
    road_mask = keep_local_lowest_layer(
        points_in_range,
        road_mask,
        position_range,
        local_lowest_cell_size,
        local_lowest_margin,
    )
    road_mask |= near_field_mask
    after_local = int(road_mask.sum())
    road_mask, profile_height_details = refine_road_by_angle_profile_height(
        points_in_range,
        road_mask,
        root_xy,
        angle_profiles,
        angle_arm_assign_half_width,
        height_consistency_below_margin,
        height_consistency_above_margin,
    )
    road_mask |= near_field_mask
    after_profile_height = int(road_mask.sum())
    road_mask, height_consistency_details = refine_road_height_consistency(
        points_in_range,
        road_mask,
        position_range,
        height_consistency_cell_size,
        height_consistency_min_cell_points,
        height_consistency_percentile,
        height_consistency_below_margin,
        height_consistency_above_margin,
        height_consistency_smooth_sigma,
    )
    road_mask |= near_field_mask
    after_height_consistency = int(road_mask.sum())

    if connected_cell_size > 0 and angle_profiles:
        arm_mask = angle_profiles_to_mask(
            angle_profiles,
            root_xy,
            position_range,
            connected_cell_size,
            angle_arm_assign_half_width,
        )
        road_mask = keep_connected_to_arm_area(
            points_in_range,
            road_mask,
            arm_mask,
            position_range,
            connected_cell_size,
            connected_min_cell_points,
        )
        road_mask |= near_field_mask

    details = {
        "raw_residual_candidates": int(raw_residual_mask.sum()),
        "strict_residual_candidates": int(strict_residual_mask.sum()),
        "profile_mode": profile_mode,
        "detected_angle_arms": int(len(angle_arms)),
        "valid_angle_arm_profiles": int(len(angle_profiles)),
        "near_field_preserved": int(near_field_mask.sum()),
        "near_field_strict_candidates": int(near_field_strict_mask.sum()),
        "near_field_raw_candidates": int(near_field_mask.sum()),
        "after_profile_filter": after_profile,
        "after_local_lowest_filter": after_local,
        "after_profile_height_filter": after_profile_height,
        "after_height_consistency_filter": after_height_consistency,
        "after_connected_filter": int(road_mask.sum()),
        "profile_height_consistency": profile_height_details,
        "height_consistency": height_consistency_details,
        "root_xy": [float(v) for v in root_xy],
        "bev_mask_cells": int(bev_mask.sum()),
        "arm_profiles": profile_details,
        "angle_arms": [
            {
                "id": int(arm["id"]),
                "angle_deg": float(arm["angle_deg"]),
                "score": float(arm["score"]),
                "support_points": int(arm["support_points"]),
            }
            for arm in angle_arms
        ],
    }
    debug = {
        "counts": counts,
        "density": density,
        "bev_mask": bev_mask,
        "angle_arms": angle_arms,
        "angle_profiles": angle_profiles,
        "angle_hist": angle_hist,
        "near_field_mask": near_field_mask,
        "root_xy": root_xy,
        "arm_cell_size": arm_cell_size,
        "position_range": position_range,
    }
    return points_in_range, residual, road_mask, details, debug


def plot_arm_debug(points, road_mask, debug, out_path, title, max_points=90000):
    plot_points, plot_mask = _sample_for_plot(points, road_mask, max_points)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.0))

    ax = axes[0]
    if np.any(~plot_mask):
        ax.scatter(
            plot_points[~plot_mask, 0],
            plot_points[~plot_mask, 1],
            s=0.25,
            c=[ROADSIDE_RGB / 255.0],
            alpha=0.45,
            linewidths=0,
        )
    if np.any(plot_mask):
        ax.scatter(
            plot_points[plot_mask, 0],
            plot_points[plot_mask, 1],
            s=0.35,
            c=[ROAD_RGB / 255.0],
            alpha=0.75,
            linewidths=0,
        )

    root = debug["root_xy"]
    angle_colors = plt.cm.Set1(np.linspace(0.0, 1.0, max(1, len(debug["angle_profiles"]))))
    for color, profile in zip(angle_colors, debug["angle_profiles"]):
        direction = profile["direction"]
        end = root + direction * profile["max_s"]
        ax.plot(
            [root[0], end[0]],
            [root[1], end[1]],
            color=color,
            linewidth=2.0,
            linestyle="--",
        )
    near = debug["near_field_mask"]
    if np.any(near):
        near_points, _ = _sample_for_plot(points[near], near[near], 12000)
        ax.scatter(
            near_points[:, 0],
            near_points[:, 1],
            s=1.2,
            c="#2557d6",
            alpha=0.55,
            linewidths=0,
        )
    ax.scatter([root[0]], [root[1]], c="black", s=28, marker="x")
    ax.set_title(title + " | angle arms")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y forward (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)

    ax = axes[1]
    xmin, ymin, _, xmax, ymax, _ = debug["position_range"]
    extent = [xmin, xmax, ymin, ymax]
    ax.imshow(
        debug["bev_mask"],
        origin="lower",
        extent=extent,
        cmap="Greys",
        alpha=0.42,
        interpolation="nearest",
    )
    for color, profile in zip(angle_colors, debug["angle_profiles"]):
        direction = profile["direction"]
        end = root + direction * profile["max_s"]
        ax.plot(
            [root[0], end[0]],
            [root[1], end[1]],
            color=color,
            linewidth=1.5,
            linestyle="--",
        )
    ax.scatter([root[0]], [root[1]], c="black", s=28, marker="x")
    ax.set_title("BEV mask + angle arms")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y forward (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _sample_for_plot(points, mask, max_points):
    if points.shape[0] <= max_points:
        return points, mask
    rng = np.random.default_rng(0)
    keep = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[keep], mask[keep]


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--lidar-processed-dir", default=None)
    parser.add_argument("--label-dir", default="data/openlane/lane3d_1000")
    parser.add_argument(
        "--position-range",
        nargs=6,
        default=[str(v) for v in DEFAULT_POSITION_RANGE],
    )
    parser.add_argument("--voxel-size", nargs=3, default=["0.2", "0.2", "1.0"])
    parser.add_argument("--lowest-k", type=int, default=5)
    parser.add_argument("--smooth-sigma-x", type=float, default=2.0)
    parser.add_argument("--smooth-sigma-y", type=float, default=2.0)
    parser.add_argument("--no-smooth-x", action="store_true")
    parser.add_argument("--no-smooth-y", action="store_true")
    parser.add_argument("--obstacle-residual-threshold", type=float, default=0.15)
    parser.add_argument("--road-below-threshold", type=float, default=0.08)
    parser.add_argument("--road-above-threshold", type=float, default=0.06)
    parser.add_argument("--local-lowest-cell-size", type=float, default=0.35)
    parser.add_argument("--local-lowest-margin", type=float, default=0.05)
    parser.add_argument("--profile-bin-size", type=float, default=1.0)
    parser.add_argument("--profile-seed-x", type=float, default=6.0)
    parser.add_argument("--profile-margin", type=float, default=0.35)
    parser.add_argument("--profile-min-seed-points", type=int, default=100)
    parser.add_argument("--arm-cell-size", type=float, default=0.35)
    parser.add_argument("--arm-density-min-cell-points", type=int, default=1)
    parser.add_argument("--arm-density-sigma", type=float, default=0.7)
    parser.add_argument("--arm-density-threshold", type=float, default=0.035)
    parser.add_argument("--arm-close-radius", type=float, default=1.0)
    parser.add_argument("--arm-open-radius", type=float, default=0.0)
    parser.add_argument("--arm-min-component-area", type=float, default=18.0)
    parser.add_argument("--arm-min-angle-separation", type=float, default=18.0)
    parser.add_argument("--arm-max-count", type=int, default=6)
    parser.add_argument("--arm-profile-min-bin-points", type=int, default=5)
    parser.add_argument("--arm-profile-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--arm-profile-min-points", type=int, default=120)
    parser.add_argument("--angle-arm-min-radius", type=float, default=6.0)
    parser.add_argument("--angle-arm-bin-deg", type=float, default=3.0)
    parser.add_argument("--angle-arm-smooth-sigma-deg", type=float, default=7.0)
    parser.add_argument("--angle-arm-min-score-ratio", type=float, default=0.18)
    parser.add_argument("--angle-arm-min-support-points", type=int, default=700)
    parser.add_argument("--angle-arm-sector-half-width", type=float, default=16.0)
    parser.add_argument("--angle-arm-assign-half-width", type=float, default=24.0)
    parser.add_argument(
        "--near-field-radius",
        type=float,
        default=6.0,
        help="Preserve strict residual candidates within this radius from the arm root. Set <=0 to disable.",
    )
    parser.add_argument(
        "--near-field-y-max",
        type=float,
        default=8.0,
        help="Preserve strict residual candidates with y <= this value. Set <=0 to disable.",
    )
    parser.add_argument(
        "--height-consistency-cell-size",
        type=float,
        default=0.5,
        help="BEV cell size for the post road-height consistency filter. Set <=0 to disable.",
    )
    parser.add_argument("--height-consistency-min-cell-points", type=int, default=3)
    parser.add_argument("--height-consistency-percentile", type=float, default=35.0)
    parser.add_argument("--height-consistency-below-margin", type=float, default=0.16)
    parser.add_argument("--height-consistency-above-margin", type=float, default=0.18)
    parser.add_argument(
        "--height-consistency-smooth-sigma",
        type=float,
        default=1.0,
        help="Meters of smoothing for the post road-height consistency surface.",
    )
    parser.add_argument(
        "--extension-bin-size",
        type=float,
        default=1.0,
        help="Longitudinal y-bin size for position-range road-surface extension.",
    )
    parser.add_argument("--extension-min-bin-points", type=int, default=25)
    parser.add_argument(
        "--extension-width-percentile",
        type=float,
        default=8.0,
        help="Legacy option retained for CLI compatibility; position-range extension ignores extracted road width.",
    )
    parser.add_argument(
        "--extension-width-padding",
        type=float,
        default=0.6,
        help="Legacy option retained for CLI compatibility; position-range extension ignores extracted road width.",
    )
    parser.add_argument(
        "--extension-min-width",
        type=float,
        default=4.0,
        help="Legacy option retained for CLI compatibility; position-range extension uses --position-range x limits.",
    )
    parser.add_argument(
        "--extension-max-width",
        type=float,
        default=16.0,
        help="Legacy option retained for CLI compatibility; position-range extension uses --position-range x limits.",
    )
    parser.add_argument(
        "--extension-smooth-sigma",
        type=float,
        default=1.2,
        help="Meters of smoothing for the generated y-direction z profile.",
    )
    parser.add_argument("--extension-extrapolate-bins", type=int, default=8)
    parser.add_argument(
        "--extension-z-clip-margin",
        type=float,
        default=20.0,
        help="Extra z margin beyond --position-range for generated extension points; prevents steep slopes from flattening at zmin/zmax.",
    )
    parser.add_argument(
        "--extension-min-forward-component",
        type=float,
        default=0.35,
        help="Legacy option retained for CLI compatibility; position-range extension does not use road-arm direction.",
    )
    parser.add_argument(
        "--connected-cell-size",
        type=float,
        default=0.0,
        help="Set >0 to keep only components connected to detected arm paths.",
    )
    parser.add_argument("--connected-min-cell-points", type=int, default=1)
    parser.add_argument("--out-dir", default="tmp/road_genetare_v3")
    return parser


def main():
    args = build_argparser().parse_args()
    position_range = parse_floats(args.position_range, 6, "position-range")
    voxel_size = parse_floats(args.voxel_size, 3, "voxel-size")
    if voxel_size[0] <= 0 or voxel_size[1] <= 0:
        raise ValueError("--voxel-size x/y must be positive.")
    if args.lowest_k <= 0:
        raise ValueError("--lowest-k must be positive.")
    if args.obstacle_residual_threshold <= 0:
        raise ValueError("--obstacle-residual-threshold must be positive.")
    if args.road_below_threshold < 0 or args.road_above_threshold < 0:
        raise ValueError("--road-below-threshold and --road-above-threshold must be non-negative.")
    if args.local_lowest_margin < 0 or args.profile_margin < 0:
        raise ValueError("--local-lowest-margin and --profile-margin must be non-negative.")
    if args.height_consistency_cell_size < 0:
        raise ValueError("--height-consistency-cell-size must be non-negative.")
    if args.height_consistency_min_cell_points <= 0:
        raise ValueError("--height-consistency-min-cell-points must be positive.")
    if not (0.0 <= args.height_consistency_percentile <= 100.0):
        raise ValueError("--height-consistency-percentile must be in [0, 100].")
    if args.height_consistency_below_margin < 0 or args.height_consistency_above_margin < 0:
        raise ValueError("--height-consistency margins must be non-negative.")
    if args.extension_bin_size <= 0 or args.extension_min_bin_points <= 0:
        raise ValueError("--extension-bin-size and --extension-min-bin-points must be positive.")
    if args.extension_z_clip_margin < 0:
        raise ValueError("--extension-z-clip-margin must be non-negative.")
    if not (0.0 <= args.extension_width_percentile < 50.0):
        raise ValueError("--extension-width-percentile must be in [0, 50).")
    if not (-1.0 <= args.extension_min_forward_component <= 1.0):
        raise ValueError("--extension-min-forward-component must be in [-1, 1].")

    repo = repo_root()
    image_input = Path(args.image_dir)
    if not image_input.is_absolute():
        image_input = (repo / image_input).resolve()
    if not image_input.exists():
        raise FileNotFoundError(f"Image path not found: {image_input}")

    lidar_root = Path(args.lidar_processed_dir) if args.lidar_processed_dir else guess_lidar_root(repo)
    if lidar_root is None or not lidar_root.exists():
        raise FileNotFoundError("LiDAR root not found. Please pass --lidar-processed-dir.")

    label_root = Path(args.label_dir)
    if not label_root.is_absolute():
        label_root = (repo / label_root).resolve()
    if not label_root.exists():
        raise FileNotFoundError(f"OpenLane label root not found: {label_root}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_input)
    if not images:
        raise FileNotFoundError(f"No image files found in: {image_input}")

    print(f"Found {len(images)} image(s)")
    print(f"LiDAR root: {lidar_root}")
    print(f"Output dir: {out_dir}")

    for image_path in images:
        try:
            seg_name, cur_key, cur_info, points = load_single_frame_points(
                image_path,
                lidar_root,
                label_root,
            )
        except Exception as exc:
            print(f"[skip] {image_path}: {exc}")
            continue

        try:
            split = find_split_name(image_path)
            segment_name = find_segment_name(image_path)
            label_segment_dir = resolve_label_segment_dir(
                label_root,
                split,
                segment_name,
                cur_key,
            )
            current_json_path = find_frame_file(label_segment_dir, cur_key, (".json",))
        except Exception as exc:
            print(f"[skip] {image_path}: {exc}")
            continue

        rotated_points = points.copy()
        rotated_points[:, :3] = rotate_xy_ccw_90(points[:, :3])
        ground_interp, _, in_range_points, occupied_cells = estimate_ground_surface(
            rotated_points,
            position_range,
            voxel_size[0],
            args.lowest_k,
            0.0 if args.no_smooth_x else args.smooth_sigma_x,
            0.0 if args.no_smooth_y else args.smooth_sigma_y,
        )
        seg_points, residual, road_mask, filter_details, debug = segment_points(
            rotated_points,
            ground_interp,
            position_range,
            args.obstacle_residual_threshold,
            args.road_below_threshold,
            args.road_above_threshold,
            args.local_lowest_cell_size,
            args.local_lowest_margin,
            args.profile_bin_size,
            args.profile_seed_x,
            args.profile_margin,
            args.profile_min_seed_points,
            args.arm_cell_size,
            args.arm_density_min_cell_points,
            args.arm_density_sigma,
            args.arm_density_threshold,
            args.arm_close_radius,
            args.arm_open_radius,
            args.arm_min_component_area,
            args.arm_min_angle_separation,
            args.arm_max_count,
            args.arm_profile_min_bin_points,
            args.arm_profile_smooth_sigma,
            args.arm_profile_min_points,
            args.angle_arm_min_radius,
            args.angle_arm_bin_deg,
            args.angle_arm_smooth_sigma_deg,
            args.angle_arm_min_score_ratio,
            args.angle_arm_min_support_points,
            args.angle_arm_sector_half_width,
            args.angle_arm_assign_half_width,
            args.near_field_radius,
            args.near_field_y_max,
            args.height_consistency_cell_size,
            args.height_consistency_min_cell_points,
            args.height_consistency_percentile,
            args.height_consistency_below_margin,
            args.height_consistency_above_margin,
            args.height_consistency_smooth_sigma,
            args.connected_cell_size,
            args.connected_min_cell_points,
        )

        image = Image.open(image_path).convert("RGB")
        extension_source = "none"
        extension_details = {}
        extended_surface = np.empty((0, 3), dtype=np.float32)
        if np.any(road_mask):
            try:
                extended_surface, extension_details = build_position_range_y_profile_surface(
                    seg_points[road_mask, :3],
                    position_range,
                    (voxel_size[0], voxel_size[1]),
                    args.extension_bin_size,
                    args.extension_min_bin_points,
                    args.extension_smooth_sigma,
                    args.extension_extrapolate_bins,
                    args.extension_z_clip_margin,
                )
                extension_source = "position_range_y_profile"
            except ValueError as exc:
                extension_details = {"mode": "none", "reason": str(exc)}
        extended_surface_vehicle = rotate_xy_cw_90(extended_surface)
        original_img_pt_fov, original_fov_mask, original_depth_fov = project_vehicle_points_to_image(
            points[:, :3],
            image,
            current_json_path,
        )
        extended_img_pt_fov, extended_fov_mask, extended_depth_fov = project_vehicle_points_to_image(
            extended_surface_vehicle,
            image,
            current_json_path,
        )
        extended_surface_fov = extended_surface[extended_fov_mask]

        rgb = colorize_segmentation(seg_points.shape[0], road_mask)
        sample_dir = out_dir / f"sample_{image_path.stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_img = sample_dir / image_path.name
        out_ply = sample_dir / f"{image_path.stem}_road_vs_roadside.ply"
        out_road_only_ply = sample_dir / f"{image_path.stem}_road_surface_only.ply"
        out_extended_ply = sample_dir / f"{image_path.stem}_road_surface_extended.ply"
        out_combined_ply = sample_dir / f"{image_path.stem}_road_vs_roadside_with_extended_fov.ply"
        out_lidar_fov_img = sample_dir / f"{image_path.stem}_lidar_fov_coverage.jpg"
        out_extended_fov_img = sample_dir / f"{image_path.stem}_road_surface_extended_fov_coverage.jpg"
        out_bev = sample_dir / f"{image_path.stem}_road_vs_roadside_bev.png"
        out_3d = sample_dir / f"{image_path.stem}_road_vs_roadside_3d.png"
        out_hist = sample_dir / f"{image_path.stem}_ground_residual_hist.png"
        out_arm_debug = sample_dir / f"{image_path.stem}_road_arm_debug.png"
        out_summary = sample_dir / f"{image_path.stem}_summary.json"

        write_ply_xyzrgb(out_ply, seg_points[:, :3], rgb)
        road_only_rgb = repeat_rgb(ROAD_RGB, int(road_mask.sum()))
        write_ply_xyzrgb(out_road_only_ply, seg_points[road_mask, :3], road_only_rgb)
        extended_rgb = repeat_rgb(EXTENDED_ROAD_RGB, extended_surface.shape[0])
        write_ply_xyzrgb(out_extended_ply, extended_surface, extended_rgb)
        combined_xyz = np.concatenate(
            [
                seg_points[~road_mask, :3],
                seg_points[road_mask, :3],
                extended_surface_fov,
            ],
            axis=0,
        )
        combined_rgb = np.concatenate(
            [
                repeat_rgb(ROADSIDE_RGB, int((~road_mask).sum())),
                repeat_rgb(ROAD_RGB, int(road_mask.sum())),
                repeat_rgb(EXTENDED_ROAD_RGB, extended_surface_fov.shape[0]),
            ],
            axis=0,
        )
        write_ply_xyzrgb(out_combined_ply, combined_xyz, combined_rgb)
        plot_bev(seg_points[:, :3], road_mask, out_bev, f"{image_path.stem} | road arm BEV")
        plot_3d(seg_points[:, :3], road_mask, out_3d, f"{image_path.stem} | road arm 3D")
        plot_residual_hist(
            residual,
            args.road_below_threshold,
            args.road_above_threshold,
            out_hist,
            f"{image_path.stem} | ground residual",
        )
        plot_arm_debug(
            seg_points[:, :3],
            road_mask,
            debug,
            out_arm_debug,
            f"{image_path.stem}",
        )
        save_lidar_fov_coverage_image(
            image,
            original_img_pt_fov,
            original_fov_mask,
            original_depth_fov,
            out_lidar_fov_img,
        )
        save_lidar_fov_coverage_image(
            image,
            extended_img_pt_fov,
            extended_fov_mask,
            extended_depth_fov,
            out_extended_fov_img,
        )
        shutil.copy2(image_path, out_img)

        summary = {
            "segment": seg_name,
            "image": str(image_path),
            "current_frame_key12": cur_key,
            "current_sample_idx": int(cur_info["sample_idx"]),
            "raw_single_frame_points": int(points.shape[0]),
            "points_inside_position_range_for_grid": int(in_range_points),
            "occupied_fitting_cells": int(occupied_cells),
            "segmented_points": int(seg_points.shape[0]),
            "road_surface_points": int(road_mask.sum()),
            "roadside_or_non_road_points": int((~road_mask).sum()),
            "extended_road_surface_points": int(extended_surface.shape[0]),
            "extended_road_surface_fov_points": int(extended_surface_fov.shape[0]),
            "extension_source": extension_source,
            "extension_details": extension_details,
            "raw_points_in_camera_fov": int(original_img_pt_fov.shape[0]),
            "road_surface_ratio": float(road_mask.mean()) if road_mask.size else 0.0,
            "position_range": [float(v) for v in position_range],
            "extension_step_xy": [float(voxel_size[0]), float(voxel_size[1])],
            "obstacle_residual_threshold": float(args.obstacle_residual_threshold),
            "road_below_threshold": float(args.road_below_threshold),
            "road_above_threshold": float(args.road_above_threshold),
            "local_lowest_cell_size": float(args.local_lowest_cell_size),
            "local_lowest_margin": float(args.local_lowest_margin),
            "profile_bin_size": float(args.profile_bin_size),
            "profile_seed_x": float(args.profile_seed_x),
            "profile_margin": float(args.profile_margin),
            "arm_cell_size": float(args.arm_cell_size),
            "arm_density_sigma": float(args.arm_density_sigma),
            "arm_density_threshold": float(args.arm_density_threshold),
            "arm_close_radius": float(args.arm_close_radius),
            "angle_arm_min_radius": float(args.angle_arm_min_radius),
            "angle_arm_bin_deg": float(args.angle_arm_bin_deg),
            "angle_arm_smooth_sigma_deg": float(args.angle_arm_smooth_sigma_deg),
            "angle_arm_min_score_ratio": float(args.angle_arm_min_score_ratio),
            "angle_arm_min_support_points": int(args.angle_arm_min_support_points),
            "angle_arm_sector_half_width": float(args.angle_arm_sector_half_width),
            "angle_arm_assign_half_width": float(args.angle_arm_assign_half_width),
            "near_field_radius": float(args.near_field_radius),
            "near_field_y_max": float(args.near_field_y_max),
            "height_consistency_cell_size": float(args.height_consistency_cell_size),
            "height_consistency_min_cell_points": int(args.height_consistency_min_cell_points),
            "height_consistency_percentile": float(args.height_consistency_percentile),
            "height_consistency_below_margin": float(args.height_consistency_below_margin),
            "height_consistency_above_margin": float(args.height_consistency_above_margin),
            "height_consistency_smooth_sigma": float(args.height_consistency_smooth_sigma),
            "extension_bin_size": float(args.extension_bin_size),
            "extension_min_bin_points": int(args.extension_min_bin_points),
            "extension_width_percentile": float(args.extension_width_percentile),
            "extension_width_padding": float(args.extension_width_padding),
            "extension_min_width": float(args.extension_min_width),
            "extension_max_width": float(args.extension_max_width),
            "extension_width_args_ignored": True,
            "extension_smooth_sigma": float(args.extension_smooth_sigma),
            "extension_extrapolate_bins": int(args.extension_extrapolate_bins),
            "extension_z_clip_margin": float(args.extension_z_clip_margin),
            "extension_min_forward_component": float(args.extension_min_forward_component),
            "connected_cell_size": float(args.connected_cell_size),
            "filter_details": filter_details,
            "voxel_size_xy": [float(voxel_size[0]), float(voxel_size[1])],
            "smooth_sigma_xy": [
                0.0 if args.no_smooth_x else float(args.smooth_sigma_x),
                0.0 if args.no_smooth_y else float(args.smooth_sigma_y),
            ],
            "outputs": [
                str(out_img),
                str(out_ply),
                str(out_road_only_ply),
                str(out_extended_ply),
                str(out_combined_ply),
                str(out_lidar_fov_img),
                str(out_extended_fov_img),
                str(out_bev),
                str(out_3d),
                str(out_hist),
                str(out_arm_debug),
            ],
        }
        out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("=== Road Segmentation V3 Preview ===")
        print("segment:", seg_name)
        print("image:", image_path)
        print("current frame key12:", cur_key)
        print("current sample_idx:", cur_info["sample_idx"])
        print("raw single-frame points:", int(points.shape[0]))
        print("segmented points:", int(seg_points.shape[0]))
        print("road surface points:", int(road_mask.sum()))
        print("roadside / non-road points:", int((~road_mask).sum()))
        print("raw FOV points:", int(original_img_pt_fov.shape[0]))
        print("extension source:", extension_source)
        print("extended road surface points:", int(extended_surface.shape[0]))
        print("extended road surface FOV points:", int(extended_surface_fov.shape[0]))
        print("road surface ratio:", f"{summary['road_surface_ratio']:.3f}")
        print("filter details:", filter_details)
        print("out image:", out_img)
        print("out extended ply:", out_extended_ply)
        print("out combined ply:", out_combined_ply)
        print("out raw fov coverage image:", out_lidar_fov_img)
        print("out extended fov coverage image:", out_extended_fov_img)
        print("out bev:", out_bev)
        print("out arm debug:", out_arm_debug)
        print("summary:", out_summary)


if __name__ == "__main__":
    main()
