"""Pair-isolated hidden-tail test of current versus guarded 4-point clustering."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.datasets.openlane import full_shape_targets as targets
from LaneLoom.tools.audit_full_shape_clustering import _actual_lanes, _clusters


def _fit_pair(short_points, donor_points, y_steps, guarded):
    lanes = [targets._LaneFit(index, points.copy(), points.copy(),
                              points.copy())
             for index, points in enumerate((short_points, donor_points))]
    step = float(np.min(np.diff(y_steps))) / 5.0
    for _ in range(20):
        groups = _clusters(lanes, 4, guarded=True) if guarded else (
            targets._cluster_lanes(lanes))
        if guarded:
            for mode_id, group in enumerate(groups):
                for lane_index in group:
                    lanes[lane_index].mode_id = mode_id
        old = (len(lanes), sum(len(lane.anchors) for lane in lanes))
        targets._filter_visible_anchors(lanes, groups, step)
        lanes = [lane for lane in lanes if len(lane.filtered) >= 4
                 and np.ptp(lane.filtered[:, 1]) >= 15.0]
        for lane in lanes:
            lane.anchors = lane.filtered.copy()
        if not any(lane.packed_index == 0 for lane in lanes):
            return None
        now = (len(lanes), sum(len(lane.anchors) for lane in lanes))
        if now == old:
            break
    groups = _clusters(lanes, 4, guarded=True) if guarded else (
        targets._cluster_lanes(lanes))
    full, _ = targets._fit_complete_lanes(lanes, groups, y_steps,
                                           float(y_steps[-1]))
    return full[next(index for index, lane in enumerate(lanes)
                     if lane.packed_index == 0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/audit.json"))
    parser.add_argument("--train-trials", type=int, default=300)
    parser.add_argument("--val-trials", type=int, default=200)
    parser.add_argument("--curated-trials", type=int, default=200)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/exact_pair_holdout.json"))
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    cfg = Config.fromfile(str(ROOT / "configs/ablations/"
                              "laneloom_openlane_image_lidar_temporal_0923.py"))
    dc = cfg.train_dataloader.dataset
    dataset = OpenLaneDataset(
        dataset_cfg=dc.dataset_cfg,
        temporal_sampling_distance=dc.temporal_sampling_distance,
        smooth_3d_gt=dc.smooth_3d_gt,
        gt_outlier_residual=dc.gt_outlier_residual,
        modality=dc.modality, split="train", max_samples=1)
    rng = random.Random(20260924)
    chosen = {}
    for group, limit in (("train", args.train_trials),
                         ("val", args.val_trials),
                         ("curated", args.curated_trials)):
        candidates = [(row, trial) for row in audit["frames"]
                      for trial in row["pseudo_short_trials"]
                      if (group == "train" and row["group"] == "random_training")
                      or (group == "val" and row["group"] == "random_validation")
                      or (group == "curated" and not row["group"].startswith(
                          "random"))]
        chosen[group] = rng.sample(candidates, min(limit, len(candidates)))
    result = defaultdict(list)
    for group, trials in chosen.items():
        for index, (row, trial) in enumerate(trials):
            label = Path(row["path"])
            packed = dataset._parse_openlane_gt(dataset._read_json(label))
            xyz = packed["gt_lanes_3d"].numpy()
            visible = packed["gt_lanes_visibility"].numpy() > .5
            lanes = _actual_lanes(xyz, visible, dataset.anchor_y_steps)
            by_id = {lane.packed_index: lane for lane in lanes}
            source = by_id[trial["source_gt_id"]]
            donor = by_id[trial["donor_gt_id"]]
            short = source.anchors[:4]
            test = source.anchors[source.anchors[:, 1] > short[-1, 1]]
            test = test[(test[:, 1] >= donor.support[0])
                        & (test[:, 1] <= donor.support[1])]
            if len(test) < 4:
                continue
            current = _fit_pair(short, donor.anchors,
                                dataset.anchor_y_steps, guarded=False)
            guarded = _fit_pair(short, donor.anchors,
                                dataset.anchor_y_steps, guarded=True)
            if current is None or guarded is None:
                continue
            idx = np.searchsorted(dataset.anchor_y_steps, test[:, 1])
            current_err = np.abs(current[idx, 2] - test[:, 2])
            guarded_err = np.abs(guarded[idx, 2] - test[:, 2])
            result[group].append(dict(
                path=str(label), source_gt_id=trial["source_gt_id"],
                donor_gt_id=trial["donor_gt_id"],
                test_points=len(test), test_end_y_m=float(test[-1, 1]),
                current_z_mae_m=float(current_err.mean()),
                guarded_z_mae_m=float(guarded_err.mean()),
                current_end_abs_m=float(current_err[-1]),
                guarded_end_abs_m=float(guarded_err[-1])))
            if (index + 1) % 100 == 0:
                print(f"{group}: {index + 1}/{len(trials)}", flush=True)
    summaries = {}
    for group, trials in result.items():
        current = np.asarray([row["current_z_mae_m"] for row in trials])
        guarded = np.asarray([row["guarded_z_mae_m"] for row in trials])
        current_end = np.asarray([row["current_end_abs_m"] for row in trials])
        guarded_end = np.asarray([row["guarded_end_abs_m"] for row in trials])
        summaries[group] = dict(
            trials=len(trials), current_mae_mean_m=float(current.mean()),
            guarded_mae_mean_m=float(guarded.mean()),
            current_mae_median_m=float(np.median(current)),
            guarded_mae_median_m=float(np.median(guarded)),
            guarded_better_fraction=float((guarded < current).mean()),
            guarded_worse_0_1m_count=int(((guarded - current) > 0.1).sum()),
            current_end_p90_m=float(np.quantile(current_end, .9)),
            guarded_end_p90_m=float(np.quantile(guarded_end, .9)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        summaries=summaries, trials=result), indent=2, ensure_ascii=False))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
