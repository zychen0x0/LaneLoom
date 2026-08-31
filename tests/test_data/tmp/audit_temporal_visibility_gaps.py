"""Audit real temporal-enhanced OpenLane GT for disconnected visibility."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmdet3d.registry import DATASETS

import LaneLoom.datasets.openlane.dataset  # noqa: F401
from LaneLoom.datasets.openlane.dataset import _timestamp_key


_DATASET = None


def _enhanced_gt(index):
    dataset = _DATASET
    current_json_path = dataset.label_files[index]
    scene_name = current_json_path.parent.name
    current_key = _timestamp_key(current_json_path)
    label_scene = dataset._load_label_scene(
        scene_name, current_json_path.parent)
    try:
        queue_keys, _, lidar_scene, label_scene = (
            dataset._sample_frame_queue_keys(
                scene_name, current_key, label_scene=label_scene))
    except KeyError:
        return None

    current_label = dataset._read_json(current_json_path)
    current_pose = np.asarray(
        current_label.get("pose", np.eye(4)),
        dtype=np.float32).reshape(4, 4)
    if lidar_scene is not None and current_key in lidar_scene["frames"]:
        current_pose = lidar_scene["frames"][current_key]["pose"]

    label_infos = []
    frame_metas = {}
    for key in queue_keys:
        json_path = label_scene["frames"].get(key)
        if json_path is None:
            return None
        label_info = dataset._read_json(json_path)
        image_path = dataset._find_image_path(label_info, scene_name, key)
        label_infos.append(label_info)
        queue_index = len(label_infos) - 1
        frame_metas[queue_index] = dataset._build_frame_meta(
            scene_name, key, label_info, image_path,
            lidar_scene, current_pose)

    current = dataset._parse_openlane_gt(label_infos[-1])
    enhanced = dataset._build_temporal_enhanced_gt(
        label_infos, frame_metas)
    return current_json_path, queue_keys, current, enhanced


def _scan_index(index):
    result = _enhanced_gt(index)
    if result is None:
        return index, []
    current_json_path, queue_keys, current, enhanced = result
    visibility = (
        enhanced["gt_lanes_visibility_enhanced"].numpy() > 0.5)
    current_visibility = current["gt_lanes_visibility"].numpy() > 0.5
    memory = enhanced["gt_lanes_memory_mask"].numpy() > 0.5
    track_ids = enhanced["gt_lane_track_ids_enhanced"].numpy()
    hits = []
    for lane_index, lane_visibility in enumerate(visibility):
        starts = lane_visibility & ~np.r_[False, lane_visibility[:-1]]
        run_count = int(starts.sum())
        if run_count < 2:
            continue
        hits.append(dict(
            sample_index=index,
            json_path=str(current_json_path),
            image_path=str(
                _DATASET._find_image_path(
                    _DATASET._read_json(current_json_path),
                    current_json_path.parent.name,
                    _timestamp_key(current_json_path))),
            queue_keys=list(queue_keys),
            lane_index=lane_index,
            track_id=int(track_ids[lane_index]),
            run_count=run_count,
            visible_anchor_indices=np.flatnonzero(
                lane_visibility).tolist(),
            visible_y=_DATASET.anchor_y_steps[
                lane_visibility].astype(float).tolist(),
            current_anchor_indices=np.flatnonzero(
                current_visibility[lane_index]).tolist(),
            memory_anchor_indices=np.flatnonzero(
                memory[lane_index]).tolist(),
        ))
    return index, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-index", type=int, action="append")
    parser.add_argument("--require-gap", action="store_true")
    parser.add_argument(
        "--output",
        default="tests/test_data/tmp/temporal_visibility_gap_audit.json")
    args = parser.parse_args()

    global _DATASET
    config = Config.fromfile(
        "configs/main/laneloom_openlane_image_lidar_Temporal.py")
    dataset_config = dict(config.train_dataloader.dataset)
    dataset_config["temporal_sampling_distance"] = (10.0, 5.0)
    _DATASET = DATASETS.build(dataset_config)

    if args.sample_index:
        sample_indices = args.sample_index
    else:
        sample_count = len(_DATASET)
        if args.max_samples is not None:
            sample_count = min(sample_count, args.max_samples)
        sample_indices = range(sample_count)
    sample_count = len(sample_indices)
    hits = []
    processed = 0
    context = mp.get_context("fork")
    with context.Pool(args.workers) as pool:
        for index, sample_hits in pool.imap_unordered(
                _scan_index, sample_indices, chunksize=32):
            processed += 1
            hits.extend(sample_hits)
            if processed % 5000 == 0:
                print(
                    f"processed={processed}/{sample_count} "
                    f"disconnected_lanes={len(hits)}",
                    flush=True)

    report = dict(
        scanned_samples=sample_count,
        disconnected_lane_count=len(hits),
        affected_sample_count=len({
            hit["sample_index"] for hit in hits}),
        hits=hits,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        key: value for key, value in report.items() if key != "hits"
    }, ensure_ascii=False), flush=True)
    if hits:
        print(json.dumps(hits[0], ensure_ascii=False), flush=True)
    if args.require_gap and not hits:
        raise SystemExit("No disconnected temporal visibility was found.")


if __name__ == "__main__":
    main()
