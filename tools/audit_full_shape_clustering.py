"""Read-only empirical audit of OpenLane full-shape template clustering.

Reports natural four-anchor pairing and pseudo-short holdout errors for the
current five-common-anchor rule versus a four-common-anchor candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.datasets.openlane import full_shape_targets as targets


def _limits(first, second):
    common = np.intersect1d(first.anchors[:, 1], second.anchors[:, 1])
    if not len(common):
        return dict(common=0, overlap_m=0.0, x_rmse=None, z_rmse=None)
    x1 = np.interp(common, first.anchors[:, 1], first.anchors[:, 0])
    x2 = np.interp(common, second.anchors[:, 1], second.anchors[:, 0])
    z1 = np.interp(common, first.anchors[:, 1], first.anchors[:, 2])
    z2 = np.interp(common, second.anchors[:, 1], second.anchors[:, 2])
    dx, dz = x1 - x2, z1 - z2
    return dict(
        common=int(len(common)),
        overlap_m=float(common[-1] - common[0]),
        x_rmse=float(np.sqrt(np.mean((dx - np.median(dx)) ** 2))),
        z_rmse=float(np.sqrt(np.mean((dz - np.median(dz)) ** 2))))


def _compatible(first, second, minimum_common, guarded=False,
                donor_min_span=0.0, donor_min_end=0.0):
    metric = _limits(first, second)
    if metric["common"] < minimum_common or metric["overlap_m"] < 8.0:
        return None
    agrees = metric["x_rmse"] <= 0.45 and metric["z_rmse"] <= 0.12
    if guarded and metric["common"] == 4:
        # Even an ineligible donor can disprove membership in a cluster.
        if not agrees:
            return False
        if len(first.anchors) == 4 and len(second.anchors) >= 8:
            short, donor = first, second
        elif len(second.anchors) == 4 and len(first.anchors) >= 8:
            short, donor = second, first
        else:
            return None
        if (donor.support[1] - short.support[1] < 15.0
                or _span(donor) < donor_min_span
                or donor.support[1] < donor_min_end):
            return None
    return agrees


def _clusters(lanes, minimum_common, guarded=False,
              donor_min_span=0.0, donor_min_end=0.0):
    order = sorted(range(len(lanes)), key=lambda index: -float(
        np.ptp(lanes[index].anchors[:, 1])))
    groups = []
    for lane_index in order:
        candidates = []
        for group_index, group in enumerate(groups):
            relations = [_compatible(lanes[lane_index], lanes[other],
                                     minimum_common, guarded,
                                     donor_min_span, donor_min_end)
                         for other in group]
            if any(value is False for value in relations):
                continue
            evidence = sum(value is True for value in relations)
            if evidence:
                candidates.append((evidence, group_index))
        if candidates:
            groups[max(candidates)[1]].append(lane_index)
        else:
            groups.append([lane_index])
    return groups


def _actual_lanes(lanes_3d, visibility, y_steps, minimum_points=4,
                  minimum_span=15.0):
    step = float(np.min(np.diff(y_steps))) / 5.0
    lanes = []
    for index in range(len(lanes_3d)):
        mask = visibility[index] & np.isfinite(lanes_3d[index]).all(axis=-1)
        anchors = np.asarray(lanes_3d[index, mask], dtype=np.float64)
        if len(anchors) < minimum_points or np.ptp(anchors[:, 1]) < minimum_span:
            continue
        lanes.append(targets._LaneFit(index, anchors, anchors.copy(),
                                      anchors.copy()))
    for _ in range(20):
        if not lanes:
            break
        groups = targets._cluster_lanes(lanes)
        old = (len(lanes), sum(len(lane.anchors) for lane in lanes))
        targets._filter_visible_anchors(lanes, groups, step)
        lanes = [lane for lane in lanes if
                 len(lane.filtered) >= minimum_points and
                 np.ptp(lane.filtered[:, 1]) >= minimum_span]
        for lane in lanes:
            lane.anchors = lane.filtered.copy()
        now = (len(lanes), sum(len(lane.anchors) for lane in lanes))
        if now == old:
            break
    return lanes


def _span(lane):
    return float(np.ptp(lane.anchors[:, 1]))


def _median_group_ids(groups, lanes):
    return [[int(lanes[index].packed_index) for index in group]
            for group in groups]


def _natural_frame(lanes):
    current = _clusters(lanes, 5)
    assert current == targets._cluster_lanes(lanes)
    relaxed = _clusters(lanes, 4)
    guarded = _clusters(lanes, 4, guarded=True)
    long_guarded = _clusters(lanes, 4, guarded=True,
                             donor_min_span=50.0, donor_min_end=60.0)
    current_lookup = {lane: index for index, group in enumerate(current)
                      for lane in group}
    relaxed_lookup = {lane: index for index, group in enumerate(relaxed)
                      for lane in group}
    guarded_lookup = {lane: index for index, group in enumerate(guarded)
                      for lane in group}
    long_guarded_lookup = {lane: index for index, group in
                           enumerate(long_guarded) for lane in group}
    natural_pairs = []
    for short in range(len(lanes)):
        if len(lanes[short].anchors) != 4:
            continue
        for other in range(len(lanes)):
            if other == short or len(lanes[other].anchors) <= 4:
                continue
            metric = _limits(lanes[short], lanes[other])
            if metric["common"] != 4 or metric["overlap_m"] < 8:
                continue
            if metric["x_rmse"] > 0.45 or metric["z_rmse"] > 0.12:
                continue
            natural_pairs.append(dict(
                short_gt_id=int(lanes[short].packed_index),
                long_gt_id=int(lanes[other].packed_index),
                short_support=[float(v) for v in lanes[short].support],
                long_support=[float(v) for v in lanes[other].support],
                beyond_short_end_m=float(lanes[other].support[1]
                                         - lanes[short].support[1]),
                x_rmse_m=metric["x_rmse"], z_rmse_m=metric["z_rmse"],
                merged_under_4=(relaxed_lookup[short] == relaxed_lookup[other]),
                merged_under_guarded4=(
                    guarded_lookup[short] == guarded_lookup[other]),
                merged_under_long_guarded4=(
                    long_guarded_lookup[short] == long_guarded_lookup[other])))
    four_ids = [index for index, lane in enumerate(lanes)
                if len(lane.anchors) == 4]
    rescued_four = sum(
        len(current[current_lookup[index]]) == 1
        and len(relaxed[relaxed_lookup[index]]) > 1
        for index in four_ids)
    guarded_rescued_four = sum(
        len(current[current_lookup[index]]) == 1
        and len(guarded[guarded_lookup[index]]) > 1
        for index in four_ids)
    long_guarded_rescued_four = sum(
        len(current[current_lookup[index]]) == 1
        and len(long_guarded[long_guarded_lookup[index]]) > 1
        for index in four_ids)
    return dict(
        eligible_lanes=len(lanes), four_anchor_lanes=len(four_ids),
        current_modes=len(current), relaxed_modes=len(relaxed),
        guarded_modes=len(guarded),
        long_guarded_modes=len(long_guarded),
        rescued_four=rescued_four,
        guarded_rescued_four=guarded_rescued_four,
        long_guarded_rescued_four=long_guarded_rescued_four,
        natural_pairs=natural_pairs,
        current_groups=_median_group_ids(current, lanes),
        relaxed_groups=_median_group_ids(relaxed, lanes),
        guarded_groups=_median_group_ids(guarded, lanes),
        long_guarded_groups=_median_group_ids(long_guarded, lanes))


def _pseudo_short_trials(lanes, y_steps):
    """Hide the tail of long GT, then compare self-fit vs compatible long GT."""
    trials = []
    for source_index, source in enumerate(lanes):
        points = source.anchors
        if len(points) < 8:
            continue
        short_points = points[:4]
        short = targets._LaneFit(source.packed_index, short_points,
                                 short_points, short_points)
        if np.ptp(short_points[:, 1]) < 15.0:
            continue
        candidates = []
        for other_index, other in enumerate(lanes):
            if source_index == other_index or len(other.anchors) < 8:
                continue
            metric = _limits(short, other)
            if (metric["common"] != 4 or metric["overlap_m"] < 8.0
                    or metric["x_rmse"] > 0.45
                    or metric["z_rmse"] > 0.12):
                continue
            test_y = points[points[:, 1] > short_points[-1, 1], 1]
            test_y = test_y[(test_y >= other.support[0])
                            & (test_y <= other.support[1])]
            if len(test_y) < 4:
                continue
            true_z = np.interp(test_y, points[:, 1], points[:, 2])
            fit_dense = targets._dense_visible(short_points,
                                               float(np.min(np.diff(y_steps))) / 5)
            own_controls = targets._fit_regularized_cubic(
                fit_dense[:, 1], fit_dense[:, 2], float(y_steps[-1]),
                (-10.0, 10.0))
            own_z = targets._bezier_basis(test_y, y_steps[-1]) @ own_controls
            other_window_z = np.interp(short_points[:, 1],
                                       other.anchors[:, 1], other.anchors[:, 2])
            offset = np.median(short_points[:, 2] - other_window_z)
            borrow_z = np.interp(test_y, other.anchors[:, 1],
                                 other.anchors[:, 2]) + offset
            candidates.append(dict(
                source_gt_id=int(source.packed_index),
                donor_gt_id=int(other.packed_index),
                source_full_span_m=_span(source),
                donor_span_m=_span(other),
                donor_end_y_m=float(other.support[1]),
                test_points=int(len(test_y)),
                test_far_y_m=float(test_y[-1]),
                x_rmse_m=metric["x_rmse"],
                z_rmse_m=metric["z_rmse"],
                own_z_mae_m=float(np.abs(own_z - true_z).mean()),
                borrowed_z_mae_m=float(np.abs(borrow_z - true_z).mean()),
                own_z_end_abs_m=float(abs(own_z[-1] - true_z[-1])),
                borrowed_z_end_abs_m=float(abs(borrow_z[-1] - true_z[-1]))))
        if candidates:
            # The hypothesis is to prefer the longest visible source.
            trials.append(max(candidates, key=lambda row: (
                row["donor_span_m"], row["donor_end_y_m"],
                -row["z_rmse_m"])))
    return trials


def _curated_paths(path, label_root):
    text = path.read_text(encoding="utf-8")
    result = []
    category = "unspecified"
    for line in text.splitlines():
        value = line.strip()
        if value.startswith("data/openlane/images/") and value.endswith(".jpg"):
            parts = Path(value).parts
            split = parts[3]
            relative = Path(*parts[4:]).with_suffix(".json")
            label = label_root / split / relative
            result.append((label, category))
        elif value and not value.startswith("#") and not value.startswith("data/"):
            category = value
    return result


def _summary(rows):
    if not rows:
        return None
    own = np.asarray([row["own_z_mae_m"] for row in rows])
    borrow = np.asarray([row["borrowed_z_mae_m"] for row in rows])
    own_end = np.asarray([row["own_z_end_abs_m"] for row in rows])
    borrow_end = np.asarray([row["borrowed_z_end_abs_m"] for row in rows])
    return dict(
        trials=len(rows),
        own_z_mae_mean_m=float(own.mean()),
        borrowed_z_mae_mean_m=float(borrow.mean()),
        own_z_mae_median_m=float(np.median(own)),
        borrowed_z_mae_median_m=float(np.median(borrow)),
        fraction_borrow_better=float((borrow < own).mean()),
        fraction_borrow_worse_by_0_3m=float(((borrow - own) > 0.3).mean()),
        own_end_abs_mean_m=float(own_end.mean()),
        borrowed_end_abs_mean_m=float(borrow_end.mean()),
        own_end_abs_p90_m=float(np.quantile(own_end, .9)),
        borrowed_end_abs_p90_m=float(np.quantile(borrow_end, .9)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="OpenLane config for the dataset being audited")
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=600)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/audit.json"))
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    dc = cfg.train_dataloader.dataset
    dataset = OpenLaneDataset(
        dataset_cfg=dc.dataset_cfg,
        temporal_sampling_distance=dc.temporal_sampling_distance,
        smooth_3d_gt=dc.smooth_3d_gt,
        gt_outlier_residual=dc.gt_outlier_residual,
        modality=dc.modality, split="train", max_samples=1)
    label_root = Path(dc.dataset_cfg.data_label_dir)
    curated = _curated_paths(ROOT / "docs/场景数据可视化.md", label_root)
    rng = random.Random(20260924)
    frames = []
    for split, count in (("training", args.train_samples),
                         ("validation", args.val_samples)):
        files = sorted((label_root / split).rglob("*.json"))[::5]
        chosen = rng.sample(files, min(count, len(files)))
        frames.extend((path, "random_" + split) for path in chosen)
    frames.extend(curated)
    seen = set()
    selected = []
    for path, group in frames:
        if path in seen:
            continue
        seen.add(path)
        selected.append((path, group))
    results = []
    counts = Counter()
    all_trials = defaultdict(list)
    for index, (path, group) in enumerate(selected):
        if not path.is_file():
            counts["missing_labels"] += 1
            continue
        packed = dataset._parse_openlane_gt(dataset._read_json(path))
        xyz = packed["gt_lanes_3d"].numpy()
        visible = packed["gt_lanes_visibility"].numpy() > .5
        lanes = _actual_lanes(xyz, visible, dataset.anchor_y_steps)
        row = _natural_frame(lanes)
        trials = _pseudo_short_trials(lanes, dataset.anchor_y_steps)
        all_trials[group].extend(trials)
        row.update(path=str(path), group=group, pseudo_short_trials=trials)
        results.append(row)
        counts[group] += 1
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(selected)}", flush=True)
    aggregations = {}
    for group in sorted(counts):
        group_rows = [row for row in results if row["group"] == group]
        pairs = [pair for row in group_rows for pair in row["natural_pairs"]]
        aggregations[group] = dict(
            frames=len(group_rows),
            eligible_lanes=sum(row["eligible_lanes"] for row in group_rows),
            four_anchor_lanes=sum(row["four_anchor_lanes"] for row in group_rows),
            current_modes=sum(row["current_modes"] for row in group_rows),
            relaxed_modes=sum(row["relaxed_modes"] for row in group_rows),
            guarded_modes=sum(row["guarded_modes"] for row in group_rows),
            long_guarded_modes=sum(
                row["long_guarded_modes"] for row in group_rows),
            rescued_four=sum(row["rescued_four"] for row in group_rows),
            guarded_rescued_four=sum(
                row["guarded_rescued_four"] for row in group_rows),
            long_guarded_rescued_four=sum(
                row["long_guarded_rescued_four"] for row in group_rows),
            natural_four_to_long_pairs=len(pairs),
            four_to_long_extends_15m=sum(
                pair["beyond_short_end_m"] >= 15 for pair in pairs),
            pseudo_short=_summary(all_trials[group]))
    report = dict(config=str(cfg.filename),
                  seed=20260924, sampling="sorted label paths [::5], random sample",
                  requested_sample_count={"train": args.train_samples,
                                          "val": args.val_samples},
                  curated_paths=len(curated),
                  aggregations=aggregations, frames=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(aggregations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
