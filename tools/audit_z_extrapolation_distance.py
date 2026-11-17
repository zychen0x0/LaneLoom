"""Measure z-target error versus metres beyond four visible GT anchors."""

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
from LaneLoom.tools.audit_full_shape_clustering import _actual_lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/full_training_audit.json"))
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--group", choices=("training", "validation"),
                        default="training")
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work_dirs/shape_clustering_audit/z_extrapolation_distance.json"))
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
    paths = [row["path"] for row in audit["frames"]
             if row["group"] == "random_" + args.group]
    rng = random.Random(20260924)
    chosen = rng.sample(paths, min(args.frames, len(paths)))
    distances = defaultdict(list)
    source_count = 0
    natural_four_spans = []
    natural_four_ends = []
    natural_four_extrapolated_anchor_counts = []
    for path in chosen:
        packed = dataset._parse_openlane_gt(dataset._read_json(Path(path)))
        xyz = packed["gt_lanes_3d"].numpy()
        visible = packed["gt_lanes_visibility"].numpy() > .5
        lanes = _actual_lanes(xyz, visible, dataset.anchor_y_steps)
        for lane in lanes:
            if len(lane.anchors) == 4:
                natural_four_spans.append(float(np.ptp(lane.anchors[:, 1])))
                natural_four_ends.append(float(lane.anchors[-1, 1]))
                natural_four_extrapolated_anchor_counts.append(int(
                    (dataset.anchor_y_steps > lane.anchors[-1, 1]).sum()))
            if len(lane.anchors) < 8:
                continue
            points = lane.anchors
            short = points[:4]
            dense = targets._dense_visible(short, 1.0)
            controls = targets._fit_regularized_cubic(
                dense[:, 1], dense[:, 2],
                float(dataset.anchor_y_steps[-1]), (-10.0, 10.0))
            hidden = points[4:]
            pred = targets._bezier_basis(
                hidden[:, 1], dataset.anchor_y_steps[-1]) @ controls
            source_count += 1
            for y, true_z, pred_z in zip(hidden[:, 1], hidden[:, 2], pred):
                ahead = float(y - short[-1, 1])
                bucket = ("0-20" if ahead <= 20 else
                          "20-40" if ahead <= 40 else
                          "40-60" if ahead <= 60 else ">60")
                distances[bucket].append(abs(float(true_z - pred_z)))
    summary = {}
    for bucket in ("0-20", "20-40", "40-60", ">60"):
        values = np.asarray(distances[bucket])
        summary[bucket] = dict(
            anchors=len(values), mean_m=float(values.mean()),
            median_m=float(np.median(values)),
            p90_m=float(np.quantile(values, .9)),
            fraction_over_0_1m=float((values > .1).mean()),
            fraction_over_0_3m=float((values > .3).mean()))
    result = dict(
        frames=len(chosen), pseudo_short_lanes=source_count,
        natural_four=dict(
            count=len(natural_four_spans),
            span_median_m=float(np.median(natural_four_spans)),
            span_p90_m=float(np.quantile(natural_four_spans, .9)),
            last_y_median_m=float(np.median(natural_four_ends)),
            last_y_p90_m=float(np.quantile(natural_four_ends, .9)),
            extrapolated_anchors_mean=float(np.mean(
                natural_four_extrapolated_anchor_counts)),
            fraction_end_le_40m=float((np.asarray(
                natural_four_ends) <= 40).mean()),
            fraction_end_le_60m=float((np.asarray(
                natural_four_ends) <= 60).mean())),
        summary=summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
