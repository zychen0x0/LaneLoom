#!/usr/bin/env python3
"""Throwaway prototype for confidence-aware 0--100 m OpenLane GT extension.

This script deliberately does not change the training target pipeline.  It tests
one candidate data transformation on named validation frames and exports the
intermediate states needed to decide whether the idea is safe enough to absorb
into ``OpenLaneDataset`` later.

Pipeline under test
-------------------
1. Sample every visible GT lane on the existing 2 m template anchors.
2. Cluster only mutually compatible, translation-aligned lane shapes.
3. Estimate a preliminary shared longitudinal z(y) shape with a per-lane offset.
4. Filter visible 2 m anchors against local x(y) and shared z(y) residuals.
5. Densify retained visible anchors fivefold (2.0 m -> 0.4 m) with PCHIP.
6. Refit a regularized cubic on [0, 100] m; share z shape inside each cluster.

The exported curves retain three evidence states: official visible support,
support borrowed from compatible lanes, and unsupported extrapolation.  Only
the first state is suitable for unchanged hard GT supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

import numpy as np
from matplotlib.lines import Line2D
from mmengine.config import Config
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear

from audit_panel_alignment import require_matplotlib_panel_alignment


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset  # noqa: E402
from LaneLoom.vis.input.visualize_sample import _load_openlane_points  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs/main/laneloom_openlane_image_lidar_temporal.py"
ANCHORS_2M = np.arange(2.0, 100.01, 2.0, dtype=np.float64)
GRID_04M = np.arange(0.0, 100.0001, 0.4, dtype=np.float64)
LANE_COLORS = (
    "#0F4D92",
    "#B64342",
    "#42949E",
    "#9A4D8E",
    "#C17C24",
    "#477A3C",
    "#6B6B9E",
    "#8C5A3C",
    "#2E7D78",
    "#7A4E7A",
)


DEFAULT_SCENES = (
    ("occlusion", "segment-7732779227944176527_2120_000_2140_000_with_camera_labels", "155917227955062700"),
    ("occlusion", "segment-7799643635310185714_680_000_700_000_with_camera_labels", "155796075957627200"),
    ("slope", "segment-17612470202990834368_2800_000_2820_000_with_camera_labels", "155364079579344600"),
    ("slope", "segment-17703234244970638241_220_000_240_000_with_camera_labels", "155269221092090800"),
    ("intersection", "segment-2367305900055174138_1881_827_1901_827_with_camera_labels", "151293026034628200"),
    ("intersection", "segment-15488266120477489949_3162_920_3182_920_with_camera_labels", "150897848610835100"),
    ("curve", "segment-366934253670232570_2229_530_2249_530_with_camera_labels", "150897516671346500"),
    ("curve", "segment-3039251927598134881_1240_610_1260_610_with_camera_labels", "150724631070026400"),
    ("branch", "segment-5373876050695013404_3817_170_3837_170_with_camera_labels", "151026502726599100"),
    ("branch", "segment-2834723872140855871_1615_000_1635_000_with_camera_labels", "150696587520602900"),
    ("branch", "segment-14811410906788672189_373_113_393_113_with_camera_labels", "150724277455060100"),
)


@dataclass
class LaneState:
    lane_id: int
    category: int
    anchors: np.ndarray
    filtered: np.ndarray
    removed: np.ndarray
    dense: np.ndarray
    mode_id: int = -1
    x_controls: np.ndarray | None = None
    z_local_controls: np.ndarray | None = None
    z_shared_controls: np.ndarray | None = None
    z_offset: float = 0.0

    @property
    def support(self) -> tuple[float, float]:
        source = self.filtered if len(self.filtered) else self.anchors
        return float(source[:, 1].min()), float(source[:, 1].max())


@dataclass
class SceneView:
    scene_type: str
    frame: str
    lanes: list[LaneState]
    bev_cloud: np.ndarray
    yz_cloud: np.ndarray


def bezier_basis(y: np.ndarray) -> np.ndarray:
    t = np.asarray(y, dtype=np.float64) / 100.0
    return np.column_stack(
        ((1.0 - t) ** 3, 3.0 * (1.0 - t) ** 2 * t,
         3.0 * (1.0 - t) * t**2, t**3)
    )


def unique_sorted(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 1] >= 2.0) & (points[:, 1] <= 100.0)]
    if len(points) == 0:
        return points.reshape(0, 3)
    order = np.argsort(points[:, 1])
    points = points[order]
    unique_y, inverse = np.unique(points[:, 1], return_inverse=True)
    if len(unique_y) == len(points):
        return points
    merged = np.zeros((len(unique_y), 3), dtype=np.float64)
    merged[:, 1] = unique_y
    for dim in (0, 2):
        sums = np.bincount(inverse, weights=points[:, dim])
        merged[:, dim] = sums / np.bincount(inverse)
    return merged


def sample_visible_anchors(points: np.ndarray) -> np.ndarray:
    points = unique_sorted(points)
    if len(points) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    mask = (ANCHORS_2M >= points[0, 1]) & (ANCHORS_2M <= points[-1, 1])
    y = ANCHORS_2M[mask]
    if len(y) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    return np.column_stack(
        (PchipInterpolator(points[:, 1], points[:, 0])(y), y,
         PchipInterpolator(points[:, 1], points[:, 2])(y))
    )


def dense_visible(points: np.ndarray) -> np.ndarray:
    points = unique_sorted(points)
    if len(points) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    y0, y1 = points[0, 1], points[-1, 1]
    y = np.arange(y0, y1 + 0.2001, 0.4)
    y = np.clip(y, y0, y1)
    y = np.unique(np.r_[y, y1])
    return np.column_stack(
        (PchipInterpolator(points[:, 1], points[:, 0])(y), y,
         PchipInterpolator(points[:, 1], points[:, 2])(y))
    )


def fit_regularized_cubic(
    y: np.ndarray,
    values: np.ndarray,
    span: float,
    bounds: tuple[float, float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a stable global cubic; short supports receive stronger curvature shrinkage."""
    design = bezier_basis(y)
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-9)
    scale = np.sqrt(weights)
    curvature = np.asarray(((1.0, -2.0, 1.0, 0.0),
                            (0.0, 1.0, -2.0, 1.0)))
    regularization = 0.002 * (100.0 / max(float(span), 8.0)) ** 2
    augmented_design = np.vstack((design * scale[:, None], math.sqrt(regularization) * curvature))
    augmented_values = np.r_[np.asarray(values) * scale, 0.0, 0.0]
    return lsq_linear(
        augmented_design,
        augmented_values,
        bounds=bounds,
        lsmr_tol="auto",
    ).x


def compatible_pair(first: LaneState, second: LaneState) -> bool | None:
    a, b = first.anchors, second.anchors
    common = np.intersect1d(a[:, 1], b[:, 1])
    if len(common) < 5 or float(common[-1] - common[0]) < 8.0:
        return None
    ax = np.interp(common, a[:, 1], a[:, 0])
    bx = np.interp(common, b[:, 1], b[:, 0])
    az = np.interp(common, a[:, 1], a[:, 2])
    bz = np.interp(common, b[:, 1], b[:, 2])
    dx = ax - bx
    dz = az - bz
    x_rmse = float(np.sqrt(np.mean((dx - np.median(dx)) ** 2)))
    z_rmse = float(np.sqrt(np.mean((dz - np.median(dz)) ** 2)))
    return x_rmse <= 0.45 and z_rmse <= 0.12


def cluster_lanes(lanes: list[LaneState]) -> list[list[int]]:
    """Complete-link-like grouping: allow unknown pairs, reject any conflicting pair."""
    order = sorted(range(len(lanes)), key=lambda i: -np.ptp(lanes[i].anchors[:, 1]))
    groups: list[list[int]] = []
    for lane_index in order:
        candidates = []
        for group_index, group in enumerate(groups):
            relations = [compatible_pair(lanes[lane_index], lanes[other]) for other in group]
            if any(relation is False for relation in relations):
                continue
            evidence = sum(relation is True for relation in relations)
            if evidence:
                candidates.append((evidence, group_index))
        if candidates:
            groups[max(candidates)[1]].append(lane_index)
        else:
            groups.append([lane_index])
    for mode_id, group in enumerate(groups):
        for lane_index in group:
            lanes[lane_index].mode_id = mode_id
    return groups


def fit_shared_profile(
    samples: dict[int, np.ndarray],
    coord: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Fit value_i(y) = shared_value_at_anchor(y) + lane_offset_i.

    Keeping one shared variable per observed 2 m anchor prevents the sharing
    stage itself from erasing a real crest, dip, or grade transition.  The
    profile is only compressed to a cubic after filtering and densification.
    """
    lane_ids = sorted(samples)
    profile_y = np.unique(np.concatenate([samples[lane_id][:, 1] for lane_id in lane_ids]))
    y_to_index = {float(y): index for index, y in enumerate(profile_y)}
    profile_size = len(profile_y)
    rows, targets, base_weights = [], [], []
    for offset_column, lane_id in enumerate(lane_ids):
        points = samples[lane_id]
        design = np.zeros((len(points), profile_size + len(lane_ids)), dtype=np.float64)
        for row, y in enumerate(points[:, 1]):
            design[row, y_to_index[float(y)]] = 1.0
        design[:, profile_size + offset_column] = 1.0
        rows.append(design)
        targets.append(points[:, coord])
        base_weights.append(np.ones(len(points), dtype=np.float64))
    design = np.vstack(rows)
    target = np.concatenate(targets)
    base_weight = np.concatenate(base_weights)
    regularizers = []
    if profile_size >= 3:
        for index in range(profile_size - 2):
            row = np.zeros(design.shape[1], dtype=np.float64)
            row[index:index + 3] = (1.0, -2.0, 1.0)
            regularizers.append(row)
    # Resolve the profile/offset gauge by fixing the longest lane's offset.
    reference_lane = max(lane_ids, key=lambda lane_id: len(samples[lane_id]))
    gauge = np.zeros(design.shape[1], dtype=np.float64)
    gauge[profile_size + lane_ids.index(reference_lane)] = 1.0
    regularization = 0.15
    solution = np.zeros(design.shape[1], dtype=np.float64)
    robust_weight = np.ones(len(target), dtype=np.float64)
    for _ in range(8):
        weight = np.sqrt(base_weight * robust_weight)
        extra_design = [math.sqrt(regularization) * row for row in regularizers]
        extra_design.append(100.0 * gauge)
        augmented_design = np.vstack((design * weight[:, None], *extra_design))
        augmented_target = np.r_[target * weight, np.zeros(len(regularizers)), 0.0]
        solution = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]
        residual = target - design @ solution
        absolute = np.abs(residual)
        robust_weight = np.minimum(1.0, delta / np.maximum(absolute, 1e-9))
    profile = solution[:profile_size]
    offsets = {
        lane_id: float(solution[profile_size + index])
        for index, lane_id in enumerate(lane_ids)
    }
    return profile_y, profile, offsets


def evaluate_profile(profile_y: np.ndarray, profile: np.ndarray, y: np.ndarray) -> np.ndarray:
    return PchipInterpolator(profile_y, profile)(np.asarray(y, dtype=np.float64))


def fit_local_support_cubic(y: np.ndarray, values: np.ndarray) -> np.ndarray:
    y0, y1 = float(y[0]), float(y[-1])
    t = (np.asarray(y) - y0) / max(y1 - y0, 1e-6)
    design = np.column_stack(
        ((1.0 - t) ** 3, 3.0 * (1.0 - t) ** 2 * t,
         3.0 * (1.0 - t) * t**2, t**3)
    )
    return np.linalg.lstsq(design, values, rcond=None)[0]


def evaluate_local_support_cubic(
    controls: np.ndarray, y_train: np.ndarray, y_query: np.ndarray
) -> np.ndarray:
    y0, y1 = float(y_train[0]), float(y_train[-1])
    t = (np.asarray(y_query) - y0) / max(y1 - y0, 1e-6)
    design = np.column_stack(
        ((1.0 - t) ** 3, 3.0 * (1.0 - t) ** 2 * t,
         3.0 * (1.0 - t) * t**2, t**3)
    )
    return design @ controls


def robust_threshold(residual: np.ndarray, floor: float, ceiling: float) -> float:
    centered = residual - np.median(residual)
    mad_sigma = 1.4826 * np.median(np.abs(centered))
    return float(np.clip(3.0 * mad_sigma, floor, ceiling))


def filter_visible_anchors(lanes: list[LaneState], groups: list[list[int]]) -> None:
    preliminary: dict[int, tuple[np.ndarray, np.ndarray, dict[int, float]]] = {}
    for mode_id, group in enumerate(groups):
        samples = {lanes[index].lane_id: lanes[index].anchors for index in group}
        preliminary[mode_id] = fit_shared_profile(samples, coord=2, delta=0.08)

    for lane in lanes:
        points = lane.anchors
        x_controls = fit_local_support_cubic(points[:, 1], points[:, 0])
        profile_y, profile, offsets = preliminary[lane.mode_id]
        x_residual = points[:, 0] - evaluate_local_support_cubic(
            x_controls, points[:, 1], points[:, 1]
        )
        z_residual = points[:, 2] - (
            evaluate_profile(profile_y, profile, points[:, 1]) + offsets[lane.lane_id]
        )
        x_threshold = robust_threshold(x_residual, floor=0.35, ceiling=0.90)
        z_threshold = robust_threshold(z_residual, floor=0.10, ceiling=0.30)
        keep = (np.abs(x_residual) <= x_threshold) & (np.abs(z_residual) <= z_threshold)
        # Do not let filtering silently shorten annotation support.
        keep[0] = True
        keep[-1] = True
        kept_span = float(np.ptp(points[keep, 1])) if int(keep.sum()) >= 2 else 0.0
        if int(keep.sum()) < 4 or kept_span < 0.8 * float(np.ptp(points[:, 1])):
            keep[:] = True
        lane.filtered = points[keep]
        lane.removed = points[~keep]
        lane.dense = dense_visible(lane.filtered)


def finalize_fits(lanes: list[LaneState], groups: list[list[int]]) -> None:
    for lane in lanes:
        span = float(np.ptp(lane.dense[:, 1]))
        lane.x_controls = fit_regularized_cubic(
            lane.dense[:, 1], lane.dense[:, 0], span, (-30.0, 30.0)
        )
        lane.z_local_controls = fit_regularized_cubic(
            lane.dense[:, 1], lane.dense[:, 2], span, (-8.0, 8.0)
        )
    for group in groups:
        samples = {lanes[index].lane_id: lanes[index].filtered for index in group}
        profile_y, profile, offsets = fit_shared_profile(samples, coord=2, delta=0.06)
        shared_y = np.arange(profile_y[0], profile_y[-1] + 0.2001, 0.4)
        shared_y = np.unique(np.r_[np.clip(shared_y, profile_y[0], profile_y[-1]), profile_y[-1]])
        shared_profile = evaluate_profile(profile_y, profile, shared_y)
        for index in group:
            lane = lanes[index]
            y0, y1 = lane.support
            supported_by_other = np.zeros(len(shared_y), dtype=bool)
            for other_index in group:
                other = lanes[other_index]
                if other.lane_id == lane.lane_id:
                    continue
                other_y0, other_y1 = other.support
                supported_by_other |= (shared_y >= other_y0) & (shared_y <= other_y1)
            borrowed = ((shared_y < y0) | (shared_y > y1)) & supported_by_other
            fit_y = np.r_[lane.dense[:, 1], shared_y[borrowed]]
            fit_z = np.r_[
                lane.dense[:, 2],
                shared_profile[borrowed] + offsets[lane.lane_id],
            ]
            fit_weight = np.r_[
                np.ones(len(lane.dense)),
                np.full(int(borrowed.sum()), 0.35),
            ]
            order = np.argsort(fit_y)
            fit_y, fit_z, fit_weight = fit_y[order], fit_z[order], fit_weight[order]
            lane.z_shared_controls = fit_regularized_cubic(
                fit_y, fit_z, float(np.ptp(fit_y)), (-8.0, 8.0), weights=fit_weight
            )
            # The lane offset has already been applied to the borrowed samples;
            # the resulting controls therefore encode absolute lane height.
            lane.z_offset = 0.0


def contiguous_runs(mask: np.ndarray) -> Iterable[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)


def plot_masked(ax, x, y, mask, **kwargs) -> None:
    for run in contiguous_runs(mask):
        if len(run) >= 2:
            ax.plot(x[run], y[run], **kwargs)


def support_count(lanes: list[LaneState], mode_id: int, y: np.ndarray) -> np.ndarray:
    count = np.zeros(len(y), dtype=np.int64)
    for lane in lanes:
        if lane.mode_id != mode_id:
            continue
        y0, y1 = lane.support
        count += (y >= y0) & (y <= y1)
    return count


def heldout_metrics(lanes: list[LaneState], groups: list[list[int]]) -> list[dict]:
    rows = []
    lane_by_id = {lane.lane_id: lane for lane in lanes}
    group_by_lane = {
        lanes[index].lane_id: group for group in groups for index in group
    }
    for target in lanes:
        points = target.filtered
        if len(points) < 8 or float(np.ptp(points[:, 1])) < 20.0:
            continue
        split = min(len(points) - 3, max(4, int(math.ceil(0.65 * len(points)))))
        train, holdout = points[:split], points[split:]
        span = float(np.ptp(train[:, 1]))
        x_local = fit_regularized_cubic(train[:, 1], train[:, 0], span, (-30.0, 30.0))
        z_local = fit_regularized_cubic(train[:, 1], train[:, 2], span, (-8.0, 8.0))
        x_pred = bezier_basis(holdout[:, 1]) @ x_local
        z_local_pred = bezier_basis(holdout[:, 1]) @ z_local

        group = group_by_lane[target.lane_id]
        shared_samples = {target.lane_id: train}
        for other_index in group:
            other = lanes[other_index]
            if other.lane_id != target.lane_id:
                shared_samples[other.lane_id] = other.filtered
        if len(shared_samples) >= 2:
            profile_y, profile, offsets = fit_shared_profile(
                shared_samples, coord=2, delta=0.06
            )
            shared_y = np.arange(profile_y[0], profile_y[-1] + 0.2001, 0.4)
            shared_y = np.unique(np.r_[np.clip(shared_y, profile_y[0], profile_y[-1]), profile_y[-1]])
            shared_z = evaluate_profile(profile_y, profile, shared_y) + offsets[target.lane_id]
            target_dense = dense_visible(train)
            supported_by_other = np.zeros(len(shared_y), dtype=bool)
            for other_index in group:
                other = lanes[other_index]
                if other.lane_id == target.lane_id:
                    continue
                other_y0, other_y1 = other.support
                supported_by_other |= (shared_y >= other_y0) & (shared_y <= other_y1)
            outside_train = (shared_y > train[-1, 1]) & supported_by_other
            fit_y = np.r_[target_dense[:, 1], shared_y[outside_train]]
            fit_z = np.r_[target_dense[:, 2], shared_z[outside_train]]
            fit_weight = np.r_[
                np.ones(len(target_dense)), np.full(int(outside_train.sum()), 0.35)
            ]
            z_shared_controls = fit_regularized_cubic(
                fit_y, fit_z, float(np.ptp(fit_y)), (-8.0, 8.0), weights=fit_weight
            )
            z_shared_pred = bezier_basis(holdout[:, 1]) @ z_shared_controls
            z_shared_rmse = float(np.sqrt(np.mean((holdout[:, 2] - z_shared_pred) ** 2)))
        else:
            z_shared_rmse = math.nan

        rows.append({
            "lane_id": target.lane_id,
            "mode_id": target.mode_id,
            "train_y_max_m": float(train[-1, 1]),
            "holdout_y_min_m": float(holdout[0, 1]),
            "holdout_points": int(len(holdout)),
            "x_local_rmse_m": float(np.sqrt(np.mean((holdout[:, 0] - x_pred) ** 2))),
            "z_local_rmse_m": float(np.sqrt(np.mean((holdout[:, 2] - z_local_pred) ** 2))),
            "z_shared_rmse_m": z_shared_rmse,
        })
    return rows


def downsample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), count, replace=False)]


def add_panel_label(
    ax,
    label: str,
    *,
    x: float = -0.03,
    y: float = 1.02,
    va: str = "bottom",
) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=8, fontweight="bold", ha="right", va=va,
    )


def render_scene(
    output_base: Path,
    scene_type: str,
    frame: str,
    lanes: list[LaneState],
    lidar: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(8.2, 4.8), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(0.58, 1.42), wspace=0.18)
    ax_bev = fig.add_subplot(grid[0, 0])
    ax_z = fig.add_subplot(grid[0, 1])

    lidar = lidar[
        np.isfinite(lidar).all(axis=1)
        & (lidar[:, 0] >= -25.0) & (lidar[:, 0] <= 25.0)
        & (lidar[:, 1] >= 0.0) & (lidar[:, 1] <= 100.0)
        & (lidar[:, 2] >= -5.0) & (lidar[:, 2] <= 5.0)
    ]
    bev_cloud = downsample(lidar, 35000, seed=int(frame[-8:]) % (2**32))
    yz_cloud = downsample(lidar[np.abs(lidar[:, 0]) <= 10.0], 24000, seed=17)
    ax_bev.scatter(
        bev_cloud[:, 0], bev_cloud[:, 1], s=0.18, c="#767676",
        alpha=0.13, linewidths=0, rasterized=True,
    )
    ax_z.scatter(
        yz_cloud[:, 1], yz_cloud[:, 2], s=0.16, c="#A8A8A8",
        alpha=0.055, linewidths=0, rasterized=True,
    )

    z_values = np.concatenate([lane.filtered[:, 2] for lane in lanes])
    z_margin = max(0.35, 0.12 * float(np.ptp(z_values)))
    z_limits = (float(z_values.min() - z_margin), float(z_values.max() + z_margin))
    ax_z.set_ylim(*z_limits)
    full_basis = bezier_basis(GRID_04M)
    for lane in lanes:
        color = LANE_COLORS[lane.lane_id % len(LANE_COLORS)]
        y0, y1 = lane.support
        own = (GRID_04M >= y0) & (GRID_04M <= y1)
        shared = (~own) & (support_count(lanes, lane.mode_id, GRID_04M) > 0)
        unsupported = ~(own | shared)
        x_full = full_basis @ lane.x_controls
        z_local = full_basis @ lane.z_local_controls
        z_shared = full_basis @ lane.z_shared_controls + lane.z_offset
        x_in_view = (x_full >= -26.0) & (x_full <= 26.0)
        local_z_in_view = (z_local >= z_limits[0]) & (z_local <= z_limits[1])
        shared_z_in_view = (z_shared >= z_limits[0]) & (z_shared <= z_limits[1])

        # BEV: x is intentionally lane-local. Outside official support it is an
        # explicit extrapolation, even if another lane supports the same y.
        plot_masked(
            ax_bev, x_full, GRID_04M, own & x_in_view,
            color=color, lw=1.45, zorder=5,
        )
        plot_masked(
            ax_bev, x_full, GRID_04M, (~own) & x_in_view, color=color, lw=1.0,
            linestyle=(0, (1.5, 2.0)), alpha=0.72, zorder=4,
        )
        ax_bev.plot(
            lane.dense[:, 0], lane.dense[:, 1], color=color, lw=0.65,
            alpha=0.55, marker=".", markersize=1.1, markevery=5, zorder=6,
        )
        ax_bev.scatter(
            lane.filtered[:, 0], lane.filtered[:, 1], s=8, facecolor="white",
            edgecolor=color, linewidth=0.55, zorder=7,
        )
        if len(lane.removed):
            ax_bev.scatter(
                lane.removed[:, 0], lane.removed[:, 1], s=15, marker="x",
                color="#E53935", linewidth=0.7, zorder=8,
            )
        # Longitudinal view: compare a lane-only second fit with the shared-z fit.
        plot_masked(
            ax_z, GRID_04M, z_local, local_z_in_view,
            color="#B64342", lw=0.65, linestyle=(0, (4.0, 2.5)),
            alpha=0.40, zorder=2,
        )
        plot_masked(
            ax_z, GRID_04M, z_shared, own & shared_z_in_view,
            color=color, lw=1.55, zorder=5,
        )
        plot_masked(
            ax_z, GRID_04M, z_shared, shared & shared_z_in_view,
            color=color, lw=1.25,
            linestyle=(0, (4.0, 2.2)), alpha=0.82, zorder=4,
        )
        plot_masked(
            ax_z, GRID_04M, z_shared, unsupported & shared_z_in_view,
            color=color, lw=0.85,
            linestyle=(0, (1.0, 2.1)), alpha=0.58, zorder=3,
        )
        ax_z.plot(
            lane.dense[:, 1], lane.dense[:, 2], color=color, lw=0.65,
            alpha=0.55, marker=".", markersize=1.1, markevery=5, zorder=6,
        )
        ax_z.scatter(
            lane.filtered[:, 1], lane.filtered[:, 2], s=8, facecolor="white",
            edgecolor=color, linewidth=0.55, zorder=7,
        )
        if len(lane.removed):
            ax_z.scatter(
                lane.removed[:, 1], lane.removed[:, 2], s=15, marker="x",
                color="#E53935", linewidth=0.7, zorder=8,
            )

    ax_bev.set(xlim=(-26, 26), ylim=(-2, 102), xlabel="Lateral x (m)", ylabel="Forward y (m)")
    ax_bev.set_aspect("equal", adjustable="box")
    mode_count = len({lane.mode_id for lane in lanes})
    ax_bev.set_title(f"BEV: lane-local x(y) · {mode_count} modes", fontsize=7.5, pad=5)
    ax_z.set(xlim=(-2, 102), xlabel="Forward y (m)", ylabel="Height z (m)")
    ax_z.set_title("Longitudinal: shared z(y) + lane offset", fontsize=7.5, pad=5)
    for axis in (ax_bev, ax_z):
        axis.grid(color="#E5E5E5", linewidth=0.45, zorder=0)
        axis.tick_params(labelsize=6, width=0.65, length=2.5)
        axis.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax_bev, "a")
    add_panel_label(ax_z, "b")

    legend = (
        Line2D([], [], color="#0F4D92", lw=1.5, label="official visible support"),
        Line2D([], [], color="#0F4D92", lw=1.3, linestyle="--", label="other-lane supported"),
        Line2D([], [], color="#0F4D92", lw=1.0, linestyle=":", label="unsupported extrapolation"),
        Line2D([], [], color="#B64342", lw=0.8, linestyle="--", label="lane-only z baseline"),
        Line2D([], [], color="#E53935", marker="x", lw=0, markersize=4, label="filtered anchor"),
    )
    fig.legend(
        handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.012),
        fontsize=5.7, frameon=False, columnspacing=1.3, handlelength=2.5,
        handletextpad=1.2,
    )
    fig.suptitle(
        f"{scene_type.capitalize()} · frame {frame} · {len(lanes)} lanes",
        fontsize=8.2, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.155, top=0.885)
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_bev, ax_z],
        panel_ids=["a", "b"],
        row_groups=[["a", "b"]],
        exemptions=[{
            "panels": ["a", "b"],
            "checks": ["panel-width"],
            "reason": "BEV preserves metric aspect while the longitudinal panel uses a readable z scale",
        }],
        json_out=str(output_base) + ".alignment.json",
        overlay_svg=str(output_base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(output_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_view_clouds(lidar: np.ndarray, frame: str) -> tuple[np.ndarray, np.ndarray]:
    lidar = lidar[
        np.isfinite(lidar).all(axis=1)
        & (lidar[:, 0] >= -25.0) & (lidar[:, 0] <= 25.0)
        & (lidar[:, 1] >= 0.0) & (lidar[:, 1] <= 100.0)
        & (lidar[:, 2] >= -5.0) & (lidar[:, 2] <= 5.0)
    ]
    bev_cloud = downsample(lidar, 35000, seed=int(frame[-8:]) % (2**32))
    yz_cloud = downsample(lidar[np.abs(lidar[:, 0]) <= 10.0], 24000, seed=17)
    return bev_cloud, yz_cloud


def style_view_axis(ax, compact: bool = False) -> None:
    ax.grid(color="#E5E5E5", linewidth=0.42, zorder=0)
    ax.tick_params(
        labelsize=5.2 if compact else 6.5,
        width=0.65,
        length=2.4,
    )
    ax.spines[["top", "right"]].set_visible(False)


def draw_bev_axis(
    ax,
    view: SceneView,
    *,
    compact: bool = False,
    title: str | None = None,
) -> None:
    ax.scatter(
        view.bev_cloud[:, 0], view.bev_cloud[:, 1],
        s=0.12 if compact else 0.22,
        c="#767676", alpha=0.12, linewidths=0, rasterized=True,
    )
    full_basis = bezier_basis(GRID_04M)
    for lane in view.lanes:
        color = LANE_COLORS[lane.lane_id % len(LANE_COLORS)]
        y0, y1 = lane.support
        own = (GRID_04M >= y0) & (GRID_04M <= y1)
        x_full = full_basis @ lane.x_controls
        in_view = (x_full >= -26.0) & (x_full <= 26.0)
        plot_masked(
            ax, x_full, GRID_04M, own & in_view,
            color=color, lw=1.25 if compact else 1.8, zorder=5,
        )
        plot_masked(
            ax, x_full, GRID_04M, (~own) & in_view,
            color=color, lw=0.85 if compact else 1.2,
            linestyle=(0, (1.3, 2.0)), alpha=0.76, zorder=4,
        )
        ax.plot(
            lane.dense[:, 0], lane.dense[:, 1], color=color,
            lw=0.42 if compact else 0.7, alpha=0.48,
            marker=".", markersize=0.8 if compact else 1.5,
            markevery=5, zorder=6,
        )
        ax.scatter(
            lane.filtered[:, 0], lane.filtered[:, 1],
            s=5 if compact else 12, facecolor="white", edgecolor=color,
            linewidth=0.5 if compact else 0.7, zorder=7,
        )
        if len(lane.removed):
            ax.scatter(
                lane.removed[:, 0], lane.removed[:, 1],
                s=10 if compact else 22, marker="x", color="#E53935",
                linewidth=0.6 if compact else 0.9, zorder=8,
            )
    ax.set(xlim=(-26, 26), ylim=(-2, 102))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Lateral x (m)" if not compact else "x (m)", fontsize=6 if compact else 7.5)
    ax.set_ylabel("Forward y (m)" if not compact else "y (m)", fontsize=6 if compact else 7.5)
    if title:
        ax.set_title(title, fontsize=6 if compact else 8.2, pad=4, fontweight="bold")
    style_view_axis(ax, compact=compact)


def draw_yz_axis(
    ax,
    view: SceneView,
    *,
    compact: bool = False,
    title: str | None = None,
) -> None:
    ax.scatter(
        view.yz_cloud[:, 1], view.yz_cloud[:, 2],
        s=0.11 if compact else 0.20,
        c="#A8A8A8", alpha=0.05, linewidths=0, rasterized=True,
    )
    z_values = np.concatenate([lane.filtered[:, 2] for lane in view.lanes])
    z_margin = max(0.35, 0.12 * float(np.ptp(z_values)))
    z_limits = (float(z_values.min() - z_margin), float(z_values.max() + z_margin))
    full_basis = bezier_basis(GRID_04M)
    for lane in view.lanes:
        color = LANE_COLORS[lane.lane_id % len(LANE_COLORS)]
        y0, y1 = lane.support
        own = (GRID_04M >= y0) & (GRID_04M <= y1)
        shared = (~own) & (support_count(view.lanes, lane.mode_id, GRID_04M) > 0)
        unsupported = ~(own | shared)
        z_local = full_basis @ lane.z_local_controls
        z_shared = full_basis @ lane.z_shared_controls + lane.z_offset
        local_in_view = (z_local >= z_limits[0]) & (z_local <= z_limits[1])
        shared_in_view = (z_shared >= z_limits[0]) & (z_shared <= z_limits[1])
        plot_masked(
            ax, GRID_04M, z_local, local_in_view,
            color="#B64342", lw=0.5 if compact else 0.75,
            linestyle=(0, (4.0, 2.5)), alpha=0.36, zorder=2,
        )
        plot_masked(
            ax, GRID_04M, z_shared, own & shared_in_view,
            color=color, lw=1.25 if compact else 1.8, zorder=5,
        )
        plot_masked(
            ax, GRID_04M, z_shared, shared & shared_in_view,
            color=color, lw=1.0 if compact else 1.5,
            linestyle=(0, (4.0, 2.2)), alpha=0.84, zorder=4,
        )
        plot_masked(
            ax, GRID_04M, z_shared, unsupported & shared_in_view,
            color=color, lw=0.7 if compact else 1.0,
            linestyle=(0, (1.0, 2.1)), alpha=0.60, zorder=3,
        )
        ax.plot(
            lane.dense[:, 1], lane.dense[:, 2], color=color,
            lw=0.42 if compact else 0.7, alpha=0.48,
            marker=".", markersize=0.8 if compact else 1.5,
            markevery=5, zorder=6,
        )
        ax.scatter(
            lane.filtered[:, 1], lane.filtered[:, 2],
            s=5 if compact else 12, facecolor="white", edgecolor=color,
            linewidth=0.5 if compact else 0.7, zorder=7,
        )
        if len(lane.removed):
            ax.scatter(
                lane.removed[:, 1], lane.removed[:, 2],
                s=10 if compact else 22, marker="x", color="#E53935",
                linewidth=0.6 if compact else 0.9, zorder=8,
            )
    ax.set(xlim=(-2, 102), ylim=z_limits)
    ax.set_xlabel("Forward y (m)" if not compact else "y (m)", fontsize=6 if compact else 7.5)
    ax.set_ylabel("Height z (m)" if not compact else "z (m)", fontsize=6 if compact else 7.5)
    if title:
        ax.set_title(title, fontsize=6 if compact else 8.2, pad=4, fontweight="bold")
    style_view_axis(ax, compact=compact)


def bev_legend_handles() -> tuple[Line2D, ...]:
    return (
        Line2D([], [], color="#0F4D92", lw=1.8, label="fit inside visible GT support"),
        Line2D([], [], color="#0F4D92", lw=1.2, linestyle=":", label="0–100 m x extrapolation"),
        Line2D([], [], color="#0F4D92", lw=0.7, marker=".", markersize=3, label="5× dense visible samples"),
        Line2D([], [], color="#0F4D92", lw=0, marker="o", markerfacecolor="white", markersize=4, label="retained 2 m GT anchor"),
        Line2D([], [], color="#E53935", marker="x", lw=0, markersize=4, label="filtered anchor"),
    )


def yz_legend_handles() -> tuple[Line2D, ...]:
    return (
        Line2D([], [], color="#0F4D92", lw=1.8, label="fit inside visible GT support"),
        Line2D([], [], color="#0F4D92", lw=1.5, linestyle="--", label="other-lane-supported z"),
        Line2D([], [], color="#0F4D92", lw=1.0, linestyle=":", label="unsupported z extrapolation"),
        Line2D([], [], color="#B64342", lw=0.8, linestyle="--", label="lane-only z baseline"),
        Line2D([], [], color="#0F4D92", lw=0.7, marker=".", markersize=3, label="5× dense visible samples"),
        Line2D([], [], color="#E53935", marker="x", lw=0, markersize=4, label="filtered anchor"),
    )


def save_single_view(fig, output_base: Path) -> None:
    fig.savefig(str(output_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_separate_views(output_root: Path, view: SceneView) -> None:
    bev_dir = output_root / "bev_xy"
    yz_dir = output_root / "yz"
    bev_dir.mkdir(parents=True, exist_ok=True)
    yz_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{view.scene_type}_{view.frame}"

    fig, ax = plt.subplots(figsize=(4.4, 7.2))
    draw_bev_axis(
        ax, view,
        title=f"BEV x–y GT extrapolation\n{view.scene_type} · {view.frame}",
    )
    fig.legend(
        handles=bev_legend_handles(), loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, 0.008), fontsize=5.6, frameon=False,
        handlelength=2.5, columnspacing=1.1, handletextpad=0.7,
    )
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.13, top=0.93)
    save_single_view(fig, bev_dir / f"{basename}_bev_xy")

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    draw_yz_axis(
        ax, view,
        title=f"Longitudinal y–z GT extrapolation · {view.scene_type} · {view.frame}",
    )
    fig.legend(
        handles=yz_legend_handles(), loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.008), fontsize=5.6, frameon=False,
        handlelength=2.5, columnspacing=1.1, handletextpad=0.7,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.90)
    save_single_view(fig, yz_dir / f"{basename}_yz")


def atlas_groups(panel_ids: list[str], columns: int) -> tuple[list[list[str]], list[list[str]]]:
    row_groups = [
        panel_ids[start:start + columns]
        for start in range(0, len(panel_ids), columns)
        if len(panel_ids[start:start + columns]) >= 2
    ]
    column_groups = []
    for column in range(columns):
        group = panel_ids[column::columns]
        if len(group) >= 2:
            column_groups.append(group)
    return row_groups, column_groups


def render_atlas(output_root: Path, views: list[SceneView], kind: str) -> None:
    if kind == "bev_xy":
        rows, columns = 4, 3
        figsize = (9.4, 14.5)
        draw = draw_bev_axis
        legend_handles = bev_legend_handles()
        title = "OpenLane GT extrapolation atlas · BEV x–y"
    else:
        rows, columns = 6, 2
        figsize = (11.2, 17.0)
        draw = draw_yz_axis
        legend_handles = yz_legend_handles()
        title = "OpenLane GT extrapolation atlas · longitudinal y–z"

    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    used_axes = []
    panel_ids = []
    for index, (axis, view) in enumerate(zip(axes.flat, views)):
        panel_id = chr(ord("a") + index)
        draw(axis, view, compact=True, title=f"{view.scene_type}\n{view.frame}")
        # Keep the panel letter on a separate baseline from the two-line title.
        # This matters most for the narrow, metric-aspect BEV panels.
        add_panel_label(axis, panel_id, x=-0.03, y=1.06, va="bottom")
        used_axes.append(axis)
        panel_ids.append(panel_id)
    for axis in list(axes.flat)[len(views):]:
        axis.set_visible(False)

    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=3 if kind == "yz" else 2,
        bbox_to_anchor=(0.5, 0.008), fontsize=6, frameon=False,
        handlelength=2.5, columnspacing=1.2, handletextpad=0.7,
    )
    fig.suptitle(title, fontsize=9.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(
        left=0.075 if kind == "yz" else 0.09,
        right=0.985,
        bottom=0.06,
        top=0.965,
        wspace=0.22 if kind == "yz" else 0.14,
        hspace=0.38 if kind == "yz" else 0.30,
    )
    row_groups, column_groups = atlas_groups(panel_ids, columns)
    output_base = output_root / f"{kind}_atlas"
    require_matplotlib_panel_alignment(
        fig,
        axes=used_axes,
        panel_ids=panel_ids,
        row_groups=row_groups,
        column_groups=column_groups,
        json_out=str(output_base) + ".alignment.json",
        overlay_svg=str(output_base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(output_base) + ".svg", bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def nanmedian(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: list[dict], holdout_rows: list[dict]) -> None:
    paired = [row for row in holdout_rows if np.isfinite(row["z_shared_rmse_m"])]
    local = np.asarray([row["z_local_rmse_m"] for row in paired])
    shared = np.asarray([row["z_shared_rmse_m"] for row in paired])
    paired_delta = local - shared
    better = int((paired_delta > 0).sum())
    anchor_total = sum(row["anchors_total"] for row in summaries)
    anchor_kept = sum(row["anchors_kept"] for row in summaries)
    lines = [
        "# GT 纵向形状共享与 0–100 m 延伸原型",
        "",
        "这是诊断原型，不是训练 GT 的正式修改。测试包含用户指定的 9 个样本，并补充了场景清单中的 2 个遮挡样本。",
        "",
        "## 结论",
        "",
        f"远端截短回放中，共有 {len(paired)} 条车道可进行本车道 z 拟合与共享 z 拟合的配对比较；共享后中位 RMSE 从 {np.median(local):.3f} m 降至 {np.median(shared):.3f} m，{better}/{len(paired)} 条改善，配对中位改善 {np.median(paired_delta):.3f} m。收益并不普遍，路口、分叉和缺少远端同组支撑的坡道存在明显回退。",
        "",
        "因此，共享区间只能作为带置信度的 z 软监督；同车道可见区间才是硬 GT；模式并集之外的 0–100 m 外推只可作为平滑先验，不能监督 visibility/existence。",
        "",
        "## 原型流程",
        "",
        "1. 把当前可见 GT 采样到原有 2 m anchors。",
        "2. 仅将重叠不少于 8 m、至少 5 个公共 anchors，且平移对齐后 x RMSE ≤ 0.45 m、z RMSE ≤ 0.12 m 的车道归为同一 mode。",
        "3. 每个 mode 拟合离散公共 z(y) 剖面和 lane-specific z offset；此阶段不压成全局三次曲线。",
        "4. 用 lane-local x(y) 和公共 z(y) 残差过滤可见 anchors，并强制保留端点与至少 80% 的原支撑跨度。",
        "5. 仅在保留的可见/共享支撑区间内用 PCHIP 从 2.0 m 加密到 0.4 m；加密点做归一化加权，不当成 5 倍独立证据。",
        "6. 二次拟合 0–100 m cubic template：x 只使用本车道点；z 使用本车道硬点和权重 0.35 的同 mode 共享软点。",
        "",
        "## 建议的监督权重",
        "",
        "- official visible：geometry weight = 1.0，visibility/existence 可监督；",
        "- other-lane supported：只监督 z，初始 weight = 0.2–0.35，禁止监督 visibility/existence；",
        "- unsupported extrapolation：geometry/visibility/existence weight = 0，仅保留曲率、坡度连续性等正则。",
        "",
        f"本轮过滤保留 {anchor_kept}/{anchor_total} 个 2 m anchors（{100.0 * anchor_kept / anchor_total:.1f}%）。LiDAR 仅为图中背景做确定性降采样，GT 拟合与回放未抽样。",
        "",
        "| Scene | Frame | Lanes | Modes | Anchors kept | x holdout RMSE | z local RMSE | z shared RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def value(key: str) -> str:
            item = row[key]
            return "n/a" if not np.isfinite(item) else f"{item:.3f} m"
        lines.append(
            f"| {row['scene_type']} | {row['frame']} | {row['lanes']} | {row['modes']} | "
            f"{row['anchors_kept']}/{row['anchors_total']} | {value('x_local_rmse_m')} | "
            f"{value('z_local_rmse_m')} | {value('z_shared_rmse_m')} |"
        )
    lines.extend(["", "## 场景图", ""])
    lines.extend([
        "- [BEV x–y 总览 PNG](views/bev_xy_atlas.png) · [PDF](views/bev_xy_atlas.pdf) · [SVG](views/bev_xy_atlas.svg)",
        "- [纵向 y–z 总览 PNG](views/yz_atlas.png) · [PDF](views/yz_atlas.pdf) · [SVG](views/yz_atlas.svg)",
        "",
    ])
    for row in summaries:
        base = f"{row['scene_type']}_{row['frame']}"
        lines.append(
            f"- {row['scene_type']} {row['frame']}: "
            f"[BEV x–y](views/bev_xy/{base}_bev_xy.png) · "
            f"[纵向 y–z](views/yz/{base}_yz.png) · "
            f"[双视图](scenes/{base}.png)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = output_dir / "scenes"
    scene_dir.mkdir(exist_ok=True)
    cfg = Config.fromfile(CONFIG_PATH)
    dataset_args = dict(cfg.val_dataloader.dataset)
    dataset_args.pop("type", None)
    dataset = OpenLaneDataset(**dataset_args)

    summaries: list[dict] = []
    all_holdout: list[dict] = []
    scene_views: list[SceneView] = []
    view_root = output_dir / "views"
    view_root.mkdir(exist_ok=True)
    for scene_type, scene, frame in DEFAULT_SCENES:
        label_path = Path(cfg.dataset_cfg.data_label_dir) / "validation" / scene / f"{frame}.json"
        image_path = Path(cfg.dataset_cfg.data_image_dir) / "validation" / scene / f"{frame}.jpg"
        if not label_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"Missing image/label pair for {scene}/{frame}")
        with label_path.open("r", encoding="utf-8") as handle:
            label = json.load(handle)
        records = dataset._parse_lane_records(label)
        lanes = []
        for lane_id, record in enumerate(records):
            anchors = sample_visible_anchors(record["ground_raw"])
            if len(anchors) < 2:
                continue
            lanes.append(LaneState(
                lane_id=lane_id,
                category=int(record["category"]),
                anchors=anchors,
                filtered=anchors.copy(),
                removed=np.zeros((0, 3), dtype=np.float64),
                dense=np.zeros((0, 3), dtype=np.float64),
            ))
        groups = cluster_lanes(lanes)
        filter_visible_anchors(lanes, groups)
        finalize_fits(lanes, groups)
        holdout = heldout_metrics(lanes, groups)
        for row in holdout:
            row.update(scene_type=scene_type, frame=frame)
        all_holdout.extend(holdout)

        lidar, _ = _load_openlane_points(cfg, scene, frame[:12])
        output_base = scene_dir / f"{scene_type}_{frame}"
        render_scene(output_base, scene_type, frame, lanes, lidar)
        bev_cloud, yz_cloud = prepare_view_clouds(lidar, frame)
        scene_view = SceneView(
            scene_type=scene_type,
            frame=frame,
            lanes=lanes,
            bev_cloud=bev_cloud,
            yz_cloud=yz_cloud,
        )
        render_separate_views(view_root, scene_view)
        scene_views.append(scene_view)
        summaries.append({
            "scene_type": scene_type,
            "scene": scene,
            "frame": frame,
            "lanes": len(lanes),
            "modes": len(groups),
            "anchors_total": int(sum(len(lane.anchors) for lane in lanes)),
            "anchors_kept": int(sum(len(lane.filtered) for lane in lanes)),
            "holdout_lanes": len(holdout),
            "x_local_rmse_m": nanmedian(holdout, "x_local_rmse_m"),
            "z_local_rmse_m": nanmedian(holdout, "z_local_rmse_m"),
            "z_shared_rmse_m": nanmedian(holdout, "z_shared_rmse_m"),
        })
        print(json.dumps(summaries[-1], ensure_ascii=False))

    render_atlas(view_root, scene_views, "bev_xy")
    render_atlas(view_root, scene_views, "yz")
    write_csv(output_dir / "scene_summary.csv", summaries)
    write_csv(output_dir / "heldout_per_lane.csv", all_holdout)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "README.md", summaries, all_holdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs/gt_shape_extension_prototype",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.output_dir.resolve())
