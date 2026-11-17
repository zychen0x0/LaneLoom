"""Build complete 0--100 m road-shape targets from visible OpenLane GT."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear


@dataclass
class _LaneFit:
    packed_index: int
    anchors: np.ndarray
    filtered: np.ndarray
    dense: np.ndarray
    mode_id: int = -1
    x_controls: np.ndarray | None = None
    z_controls: np.ndarray | None = None

    @property
    def support(self):
        points = self.filtered if len(self.filtered) else self.anchors
        return float(points[0, 1]), float(points[-1, 1])


def _bezier_basis(y, length):
    t = np.asarray(y, dtype=np.float64) / float(length)
    return np.column_stack((
        (1.0 - t) ** 3,
        3.0 * (1.0 - t) ** 2 * t,
        3.0 * (1.0 - t) * t ** 2,
        t ** 3,
    ))


def _dense_visible(points, step):
    y0, y1 = float(points[0, 1]), float(points[-1, 1])
    sample_y = np.arange(y0, y1 + step * 0.5, step)
    sample_y = np.unique(np.r_[np.clip(sample_y, y0, y1), y1])
    return np.column_stack((
        PchipInterpolator(points[:, 1], points[:, 0])(sample_y),
        sample_y,
        PchipInterpolator(points[:, 1], points[:, 2])(sample_y),
    ))


def _fit_regularized_cubic(y, values, length, bounds, weights=None,
                           visible_fit_tolerance_m=None):
    design = _bezier_basis(y, length)
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-9)
    scale = np.sqrt(weights)
    curvature = np.asarray((
        (1.0, -2.0, 1.0, 0.0),
        (0.0, 1.0, -2.0, 1.0),
    ))
    support_span = max(float(np.ptp(y)), 8.0)
    regularization = 0.002 * (float(length) / support_span) ** 2
    weighted_design = design * scale[:, None]
    weighted_values = np.asarray(values, dtype=np.float64) * scale
    reference = None
    if visible_fit_tolerance_m is not None:
        if visible_fit_tolerance_m <= 0:
            raise ValueError('visible_fit_tolerance_m must be positive.')
        reference = lsq_linear(
            weighted_design, weighted_values, bounds=bounds,
            lsmr_tol='auto').x
    # Preserve the original fit by default for existing experiment configs.
    # In the fidelity-constrained variant, smoothness may stabilize unsupported
    # extrapolation only while retaining the unregularized visible prediction.
    for _ in range(12 if reference is not None else 1):
        augmented_design = np.vstack((
            weighted_design, math.sqrt(regularization) * curvature))
        augmented_values = np.r_[weighted_values, 0.0, 0.0]
        fitted = lsq_linear(
            augmented_design, augmented_values, bounds=bounds,
            lsmr_tol='auto').x
        if (reference is None or np.max(np.abs(
                design @ (fitted - reference))) <= visible_fit_tolerance_m):
            return fitted
        regularization *= 0.1
    return reference


def _fit_local_support(y, values):
    y0, y1 = float(y[0]), float(y[-1])
    t = (np.asarray(y) - y0) / max(y1 - y0, 1e-6)
    design = np.column_stack((
        (1.0 - t) ** 3,
        3.0 * (1.0 - t) ** 2 * t,
        3.0 * (1.0 - t) * t ** 2,
        t ** 3,
    ))
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _evaluate_local_support(controls, train_y, query_y):
    y0, y1 = float(train_y[0]), float(train_y[-1])
    t = (np.asarray(query_y) - y0) / max(y1 - y0, 1e-6)
    design = np.column_stack((
        (1.0 - t) ** 3,
        3.0 * (1.0 - t) ** 2 * t,
        3.0 * (1.0 - t) * t ** 2,
        t ** 3,
    ))
    return design @ controls


def _compatible(first, second, guarded_short_merge=False):
    common = np.intersect1d(first.anchors[:, 1], second.anchors[:, 1])
    if len(common) < (4 if guarded_short_merge else 5):
        return None
    if float(common[-1] - common[0]) < 8.0:
        return None
    first_x = np.interp(common, first.anchors[:, 1], first.anchors[:, 0])
    second_x = np.interp(common, second.anchors[:, 1], second.anchors[:, 0])
    first_z = np.interp(common, first.anchors[:, 1], first.anchors[:, 2])
    second_z = np.interp(common, second.anchors[:, 1], second.anchors[:, 2])
    delta_x = first_x - second_x
    delta_z = first_z - second_z
    x_rmse = np.sqrt(np.mean((delta_x - np.median(delta_x)) ** 2))
    z_rmse = np.sqrt(np.mean((delta_z - np.median(delta_z)) ** 2))
    compatible = bool(x_rmse <= 0.45 and z_rmse <= 0.12)
    if len(common) >= 5 or not guarded_short_merge or not compatible:
        return compatible
    if len(first.anchors) == 4 and len(second.anchors) >= 8:
        short, donor = first, second
    elif len(second.anchors) == 4 and len(first.anchors) >= 8:
        short, donor = second, first
    else:
        return None
    if (donor.support[1] - donor.support[0] < 50.0
            or donor.support[1] < 60.0
            or donor.support[1] - short.support[1] < 15.0):
        return None
    return True


def _cluster_lanes(lanes, guarded_short_merge=False):
    order = sorted(
        range(len(lanes)),
        key=lambda index: -float(np.ptp(lanes[index].anchors[:, 1])))
    groups = []
    for lane_index in order:
        candidates = []
        for group_index, group in enumerate(groups):
            relations = [
                _compatible(lanes[lane_index], lanes[other],
                            guarded_short_merge=guarded_short_merge)
                for other in group]
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


def _z_confidence(anchor_y, lanes, group, grace_m, decay_m):
    distance = np.full(len(anchor_y), np.inf, dtype=np.float64)
    for index in group:
        low, high = lanes[index].support
        distance = np.minimum(distance, np.maximum(
            np.maximum(low - anchor_y, anchor_y - high), 0.0))
    return np.exp(-np.maximum(distance - grace_m, 0.0) / decay_m).astype(
        np.float32)


def _fit_shared_profile(lanes, group, delta, use_filtered=False):
    def lane_points(index):
        return lanes[index].filtered if use_filtered else lanes[index].anchors

    profile_y = np.unique(np.concatenate([
        lane_points(index)[:, 1] for index in group]))
    y_to_index = {float(y): index for index, y in enumerate(profile_y)}
    profile_size = len(profile_y)
    rows = []
    targets = []
    for offset_column, lane_index in enumerate(group):
        points = lane_points(lane_index)
        design = np.zeros(
            (len(points), profile_size + len(group)), dtype=np.float64)
        for row, y in enumerate(points[:, 1]):
            design[row, y_to_index[float(y)]] = 1.0
        design[:, profile_size + offset_column] = 1.0
        rows.append(design)
        targets.append(points[:, 2])
    design = np.vstack(rows)
    target = np.concatenate(targets)
    regularizers = []
    for index in range(max(profile_size - 2, 0)):
        row = np.zeros(design.shape[1], dtype=np.float64)
        row[index:index + 3] = (1.0, -2.0, 1.0)
        regularizers.append(row)
    reference = max(group, key=lambda index: len(lane_points(index)))
    gauge = np.zeros(design.shape[1], dtype=np.float64)
    gauge[profile_size + group.index(reference)] = 1.0
    robust_weight = np.ones(len(target), dtype=np.float64)
    solution = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(8):
        scale = np.sqrt(robust_weight)
        extras = [math.sqrt(0.15) * row for row in regularizers]
        extras.append(100.0 * gauge)
        augmented_design = np.vstack((design * scale[:, None], *extras))
        augmented_target = np.r_[
            target * scale, np.zeros(len(regularizers)), 0.0]
        solution = np.linalg.lstsq(
            augmented_design, augmented_target, rcond=None)[0]
        residual = np.abs(target - design @ solution)
        robust_weight = np.minimum(1.0, delta / np.maximum(residual, 1e-9))
    offsets = {
        lane_index: float(solution[profile_size + offset_column])
        for offset_column, lane_index in enumerate(group)
    }
    return profile_y, solution[:profile_size], offsets


def _robust_threshold(residual, floor, ceiling):
    centered = residual - np.median(residual)
    sigma = 1.4826 * np.median(np.abs(centered))
    return float(np.clip(3.0 * sigma, floor, ceiling))


def _filter_visible_anchors(lanes, groups, dense_step):
    shared_profiles = {
        mode_id: _fit_shared_profile(lanes, group, delta=0.08)
        for mode_id, group in enumerate(groups)
    }
    for lane_index, lane in enumerate(lanes):
        points = lane.anchors
        x_controls = _fit_local_support(points[:, 1], points[:, 0])
        profile_y, profile, offsets = shared_profiles[lane.mode_id]
        x_residual = points[:, 0] - _evaluate_local_support(
            x_controls, points[:, 1], points[:, 1])
        shared_z = PchipInterpolator(profile_y, profile)(points[:, 1])
        z_residual = points[:, 2] - (
            shared_z + offsets[lane_index])
        keep = (
            (np.abs(x_residual) <= _robust_threshold(
                x_residual, 0.35, 0.90))
            & (np.abs(z_residual) <= _robust_threshold(
                z_residual, 0.10, 0.30)))
        if keep.sum() < min(4, len(points)):
            normalized_residual = (
                np.abs(x_residual) / max(
                    _robust_threshold(x_residual, 0.35, 0.90), 1e-6)
                + np.abs(z_residual) / max(
                    _robust_threshold(z_residual, 0.10, 0.30), 1e-6))
            keep[:] = False
            keep[np.argsort(normalized_residual)[:min(4, len(points))]] = True
        lane.filtered = points[keep]
        lane.dense = _dense_visible(lane.filtered, dense_step)


def _fit_complete_lanes(lanes, groups, anchor_y, length,
                        visible_fit_tolerance_m=None):
    basis = _bezier_basis(anchor_y, length)
    origin_basis = _bezier_basis([0.0], length)[0]
    full_curves = {}
    relative_curves = {}
    for lane in lanes:
        lane.x_controls = _fit_regularized_cubic(
            lane.dense[:, 1], lane.dense[:, 0], length, (-50.0, 50.0),
            visible_fit_tolerance_m=visible_fit_tolerance_m)

    for group in groups:
        profile_y, profile, offsets = _fit_shared_profile(
            lanes, group, delta=0.06, use_filtered=True)
        shared_step = max(float(np.min(np.diff(anchor_y))) / 5.0, 0.1)
        shared_y = np.arange(
            profile_y[0], profile_y[-1] + shared_step * 0.5, shared_step)
        shared_y = np.unique(np.r_[
            np.clip(shared_y, profile_y[0], profile_y[-1]), profile_y[-1]])
        shared_z = PchipInterpolator(profile_y, profile)(shared_y)
        for lane_index in group:
            lane = lanes[lane_index]
            own_start, own_end = lane.support
            supported_by_other = np.zeros(len(shared_y), dtype=bool)
            for other_index in group:
                if other_index == lane_index:
                    continue
                other_start, other_end = lanes[other_index].support
                supported_by_other |= (
                    (shared_y >= other_start) & (shared_y <= other_end))
            borrowed = (
                ((shared_y < own_start) | (shared_y > own_end))
                & supported_by_other)
            fit_y = np.r_[lane.dense[:, 1], shared_y[borrowed]]
            fit_z = np.r_[
                lane.dense[:, 2],
                shared_z[borrowed] + offsets[lane_index],
            ]
            fit_weight = np.r_[
                np.ones(len(lane.dense)),
                np.full(int(borrowed.sum()), 0.35),
            ]
            order = np.argsort(fit_y)
            lane.z_controls = _fit_regularized_cubic(
                fit_y[order], fit_z[order], length, (-10.0, 10.0),
                weights=fit_weight[order],
                visible_fit_tolerance_m=visible_fit_tolerance_m)
            x = basis @ lane.x_controls
            z = basis @ lane.z_controls
            curve = np.column_stack((x, anchor_y, z))
            full_curves[lane_index] = curve
            relative = curve.copy()
            relative[:, 0] -= origin_basis @ lane.x_controls
            relative[:, 2] -= origin_basis @ lane.z_controls
            relative_curves[lane_index] = relative
    return full_curves, relative_curves


def build_full_shape_targets(
        lanes_3d,
        visibility,
        anchor_y_steps,
        max_templates=5,
        min_lane_points=4,
        min_lane_span=15.0,
        guarded_short_merge=False,
        z_extrapolation_grace_m=20.0,
        z_extrapolation_decay_m=None,
        visible_fit_tolerance_m=None):
    """Return full per-lane curves and a fixed-size shape target dictionary.

    Visible GT is filtered and densified fivefold before a regularized cubic
    is fit over the complete model range. Compatible lanes share longitudinal
    height evidence. The exported mode templates remove x/z translation so
    they match ``LiDARShapeTokenPrior``'s origin-relative representation.
    The dictionary always has ``max_templates`` slots, but only modes supported
    by observed geometry are marked valid. Unsupported slots remain free to
    learn other shapes instead of being collapsed onto duplicated targets.
    """
    lanes_3d = np.asarray(lanes_3d, dtype=np.float64)
    visibility = np.asarray(visibility) > 0.5
    anchor_y = np.asarray(anchor_y_steps, dtype=np.float64)
    if lanes_3d.ndim != 3 or lanes_3d.shape[:2] != visibility.shape:
        raise ValueError('lanes_3d and visibility shapes are inconsistent.')
    if lanes_3d.shape[1] != len(anchor_y):
        raise ValueError('anchor_y_steps must match the lane anchor dimension.')
    if len(anchor_y) < 2 or np.any(np.diff(anchor_y) <= 0):
        raise ValueError(
            'anchor_y_steps must contain at least two increasing values.')
    if max_templates <= 0:
        raise ValueError('max_templates must be positive.')
    if min_lane_points < 2:
        raise ValueError('min_lane_points must be at least two.')
    if min_lane_span < 0:
        raise ValueError('min_lane_span must be non-negative.')
    if z_extrapolation_grace_m < 0:
        raise ValueError('z_extrapolation_grace_m must be non-negative.')
    if (z_extrapolation_decay_m is not None
            and z_extrapolation_decay_m <= 0):
        raise ValueError('z_extrapolation_decay_m must be positive.')
    if visible_fit_tolerance_m is not None and visible_fit_tolerance_m <= 0:
        raise ValueError('visible_fit_tolerance_m must be positive.')
    length = float(anchor_y[-1])
    dense_step = float(np.min(np.diff(anchor_y))) / 5.0
    lanes = []
    for packed_index in range(len(lanes_3d)):
        valid = visibility[packed_index] & np.isfinite(
            lanes_3d[packed_index]).all(axis=-1)
        anchors = lanes_3d[packed_index, valid]
        if (len(anchors) < min_lane_points
                or float(np.ptp(anchors[:, 1])) < min_lane_span):
            continue
        lanes.append(_LaneFit(
            packed_index=packed_index,
            anchors=anchors,
            filtered=anchors.copy(),
            dense=anchors.copy(),
        ))

    full_lanes = np.zeros_like(lanes_3d, dtype=np.float32)
    full_valid = np.zeros(lanes_3d.shape[0], dtype=bool)
    shape_templates = np.zeros(
        (max_templates, len(anchor_y), 3), dtype=np.float32)
    template_valid = np.zeros(max_templates, dtype=bool)
    use_z_confidence = z_extrapolation_decay_m is not None
    lane_z_weights = np.zeros(visibility.shape, dtype=np.float32)
    template_z_weights = np.zeros(
        (max_templates, len(anchor_y)), dtype=np.float32)

    def output():
        result = dict(
            gt_lanes_3d_full=full_lanes,
            gt_lanes_full_valid=full_valid,
            gt_shape_templates=shape_templates,
            gt_shape_template_valid=template_valid)
        if use_z_confidence:
            result.update(
                gt_lanes_full_z_weights=lane_z_weights,
                gt_shape_template_z_weights=template_z_weights)
        return result

    if not lanes:
        return output()

    # Outlier removal can shorten a lane below the support threshold. Repeat
    # grouping/filtering until no more points or lanes are removed, so every
    # exported hard target satisfies the threshold on the evidence actually
    # used by the fit (rather than only on the raw visible anchors).
    while lanes:
        groups = _cluster_lanes(lanes, guarded_short_merge)
        previous_lane_count = len(lanes)
        previous_point_count = sum(len(lane.anchors) for lane in lanes)
        _filter_visible_anchors(lanes, groups, dense_step)
        lanes = [
            lane for lane in lanes
            if (len(lane.filtered) >= min_lane_points
                and float(np.ptp(lane.filtered[:, 1])) >= min_lane_span)
        ]
        for lane in lanes:
            lane.anchors = lane.filtered.copy()
        current_point_count = sum(len(lane.anchors) for lane in lanes)
        if (len(lanes) == previous_lane_count
                and current_point_count == previous_point_count):
            break
    if not lanes:
        return output()
    groups = _cluster_lanes(lanes, guarded_short_merge)
    full_curves, relative_curves = _fit_complete_lanes(
        lanes, groups, anchor_y, length, visible_fit_tolerance_m)
    for lane_index, lane in enumerate(lanes):
        full_lanes[lane.packed_index] = full_curves[lane_index]
        full_valid[lane.packed_index] = True
    if use_z_confidence:
        for group in groups:
            weights = _z_confidence(
                anchor_y, lanes, group, z_extrapolation_grace_m,
                z_extrapolation_decay_m)
            for lane_index in group:
                lane_z_weights[lanes[lane_index].packed_index] = weights

    ranked_groups = sorted(
        groups,
        key=lambda group: -sum(
            float(np.ptp(lanes[index].filtered[:, 1])) for index in group))
    for template_index, group in enumerate(ranked_groups[:max_templates]):
        shape_templates[template_index] = np.median(np.stack([
            relative_curves[index] for index in group]), axis=0)
        template_valid[template_index] = True
        if use_z_confidence:
            template_z_weights[template_index] = _z_confidence(
                anchor_y, lanes, group, z_extrapolation_grace_m,
                z_extrapolation_decay_m)
    return output()
