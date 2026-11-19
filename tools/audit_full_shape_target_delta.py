"""Compare actual full-shape targets under current vs conservative 4-point clustering."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.datasets.openlane import full_shape_targets as targets
from LaneLoom.tools.audit_full_shape_clustering import _clusters


def _guarded_clusters(lanes):
    groups = _clusters(lanes, 4, guarded=True,
                       donor_min_span=50.0, donor_min_end=60.0)
    for mode_id, group in enumerate(groups):
        for lane_index in group:
            lanes[lane_index].mode_id = mode_id
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="OpenLane config used to generate the audit")
    parser.add_argument("--audit", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/audit.json"))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/target_delta.json"))
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    cfg = Config.fromfile(args.config)
    dc = cfg.train_dataloader.dataset
    dataset = OpenLaneDataset(
        dataset_cfg=dc.dataset_cfg,
        temporal_sampling_distance=dc.temporal_sampling_distance,
        smooth_3d_gt=dc.smooth_3d_gt,
        gt_outlier_residual=dc.gt_outlier_residual,
        modality=dc.modality, split="train", max_samples=1)
    rows = []
    original_cluster = targets._cluster_lanes
    for index, frame in enumerate(audit["frames"]):
        if frame["current_groups"] == frame["long_guarded_groups"]:
            continue
        packed = dataset._parse_openlane_gt(dataset._read_json(Path(frame["path"])))
        xyz = packed["gt_lanes_3d"].numpy()
        visibility = packed["gt_lanes_visibility"].numpy() > .5
        options = dict(max_templates=5, min_lane_points=4,
                       min_lane_span=15.0)
        current = targets.build_full_shape_targets(
            xyz, visibility, dataset.anchor_y_steps, **options)
        try:
            targets._cluster_lanes = _guarded_clusters
            guarded = targets.build_full_shape_targets(
                xyz, visibility, dataset.anchor_y_steps, **options)
        finally:
            targets._cluster_lanes = original_cluster
        for lane_id in range(len(xyz)):
            if not (current["gt_lanes_full_valid"][lane_id]
                    and guarded["gt_lanes_full_valid"][lane_id]):
                continue
            mask = visibility[lane_id]
            current_curve = current["gt_lanes_3d_full"][lane_id]
            guarded_curve = guarded["gt_lanes_3d_full"][lane_id]
            true_curve = xyz[lane_id]
            rows.append(dict(
                path=frame["path"], group=frame["group"], lane_id=lane_id,
                visible_count=int(mask.sum()),
                current_template_count=int(
                    current["gt_shape_template_valid"].sum()),
                guarded_template_count=int(
                    guarded["gt_shape_template_valid"].sum()),
                delta_z100_m=float(guarded_curve[-1, 2]
                                   - current_curve[-1, 2]),
                delta_x100_m=float(guarded_curve[-1, 0]
                                   - current_curve[-1, 0]),
                current_visible_z_mae_m=float(np.abs(
                    current_curve[mask, 2] - true_curve[mask, 2]).mean()),
                guarded_visible_z_mae_m=float(np.abs(
                    guarded_curve[mask, 2] - true_curve[mask, 2]).mean())))
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(audit['frames'])}", flush=True)
    summary = {}
    for group, selector in (("train", lambda row: row["group"] == "random_training"),
                            ("val", lambda row: row["group"] == "random_validation"),
                            ("curated", lambda row: not row["group"].startswith("random"))):
        selected = [row for row in rows if selector(row)]
        if not selected:
            continue
        for bucket, values in (("four", [row for row in selected
                                           if row["visible_count"] == 4]),
                               ("all", selected)):
            if not values:
                continue
            dz = np.asarray([abs(row["delta_z100_m"]) for row in values])
            dx = np.asarray([abs(row["delta_x100_m"]) for row in values])
            old = np.asarray([row["current_visible_z_mae_m"]
                              for row in values])
            new = np.asarray([row["guarded_visible_z_mae_m"]
                              for row in values])
            summary[f"{group}_{bucket}"] = dict(
                lanes=len(values),
                z100_changed_over_0_3m=int((dz > .3).sum()),
                z100_changed_over_1m=int((dz > 1).sum()),
                z100_abs_delta_p50_m=float(np.median(dz)),
                z100_abs_delta_p90_m=float(np.quantile(dz, .9)),
                x100_abs_delta_p90_m=float(np.quantile(dx, .9)),
                current_visible_z_mae_mean_m=float(old.mean()),
                guarded_visible_z_mae_mean_m=float(new.mean()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, lanes=rows),
                                      ensure_ascii=False, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
