#!/usr/bin/env python3
"""Shared ground-truth visualization for OpenLane and ONCE samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
for import_root in (WORKSPACE_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from LaneLoom.datasets.openlane.dataset import (  # noqa: E402
    _build_cam_to_ground_lane,
)
from LaneLoom.datasets.once.dataset import CAMERA_TO_GROUND  # noqa: E402
from vis.input.visualize_sample import (  # noqa: E402
    _draw_gt_overlay,
    _load_configs,
    _load_json,
    _project_gt_lanes,
    _resolve_label,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
BEV_SIZE = (800, 1000)
LANE_COLORS = (
    (22, 163, 74),
    (234, 88, 12),
    (2, 132, 199),
    (219, 39, 119),
    (124, 58, 237),
)


def _ground_lanes(label, dataset_name):
    if dataset_name == 'once':
        lanes = []
        for points in label.get('lanes', []):
            lane_camera = np.asarray(points, dtype=np.float32)
            if lane_camera.ndim != 2 or lane_camera.shape[1] != 3:
                continue
            lane_h = np.concatenate([
                lane_camera,
                np.ones((lane_camera.shape[0], 1), dtype=np.float32),
            ], axis=1)
            lanes.append((CAMERA_TO_GROUND @ lane_h.T).T[:, :3])
        return lanes

    lane_cam_to_ground = _build_cam_to_ground_lane(label['extrinsic'])
    lanes = []
    for lane_info in label.get('lane_lines', []):
        lane_camera = np.asarray(lane_info.get('xyz', []), dtype=np.float32)
        if lane_camera.ndim != 2 or lane_camera.shape[0] != 3:
            continue
        visibility = np.asarray(
            lane_info.get('visibility', np.ones(lane_camera.shape[1])),
            dtype=np.float32,
        ) > 0
        lane_h = np.vstack([
            lane_camera,
            np.ones((1, lane_camera.shape[1]), dtype=np.float32),
        ])
        lane_ground = (lane_cam_to_ground @ lane_h)[:3].T
        lanes.append(lane_ground[visibility])
    return lanes


def _draw_bev(lanes, point_cloud_range, size=BEV_SIZE):
    width, height = size
    x_min, y_min, _, x_max, y_max, _ = map(float, point_cloud_range)
    canvas = Image.new('RGB', size, (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    def to_pixel(x, y):
        u = (x - x_min) / (x_max - x_min) * (width - 1)
        v = (y_max - y) / (y_max - y_min) * (height - 1)
        return int(round(u)), int(round(v))

    grid_step = 10
    first_x = int(np.ceil(x_min / grid_step) * grid_step)
    first_y = int(np.ceil(y_min / grid_step) * grid_step)
    for x in range(first_x, int(np.floor(x_max)) + 1, grid_step):
        u, _ = to_pixel(x, y_min)
        draw.line((u, 0, u, height - 1), fill=(215, 222, 230), width=1)
    for y in range(first_y, int(np.floor(y_max)) + 1, grid_step):
        _, v = to_pixel(x_min, y)
        draw.line((0, v, width - 1, v), fill=(215, 222, 230), width=1)
    if x_min <= 0 <= x_max:
        center_u, _ = to_pixel(0, y_min)
        draw.line(
            (center_u, 0, center_u, height - 1),
            fill=(100, 116, 139),
            width=2,
        )

    for lane_index, lane in enumerate(lanes):
        lane = np.asarray(lane, dtype=np.float32)
        if lane.ndim != 2 or lane.shape[1] < 2:
            continue
        finite = np.isfinite(lane[:, :2]).all(axis=1)
        points = [to_pixel(x, y) for x, y in lane[finite, :2]]
        if len(points) >= 2:
            draw.line(
                points,
                fill=LANE_COLORS[lane_index % len(LANE_COLORS)],
                width=5,
            )
        for u, v in points:
            draw.ellipse(
                (u - 3, v - 3, u + 3, v + 3),
                fill=LANE_COLORS[lane_index % len(LANE_COLORS)],
            )
    return canvas


def visualize_gt(image_path, dataset_name, output_root=DEFAULT_OUTPUT_ROOT):
    if dataset_name not in {'openlane', 'once'}:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'Input image does not exist: {image_path}')

    cfg = _load_configs()[dataset_name]
    label_path, scene_name, frame_key = _resolve_label(
        image_path, dataset_name, cfg)
    output_dir = (
        Path(output_root).expanduser().resolve()
        / dataset_name / scene_name / frame_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert('RGB')
    label = _load_json(label_path)
    overlay_path = output_dir / '01_gt_overlay.jpg'
    bev_path = output_dir / '02_gt_bev.jpg'
    _draw_gt_overlay(
        image, _project_gt_lanes(label, dataset_name, cfg)
    ).save(overlay_path, quality=95)
    lanes = _ground_lanes(label, dataset_name)
    _draw_bev(lanes, cfg.point_cloud_range).save(bev_path, quality=95)

    report = {
        'dataset': dataset_name,
        'scene': scene_name,
        'sample': frame_key,
        'source_image': str(image_path),
        'source_label': str(label_path),
        'lane_count': len(lanes),
        'output_dir': str(output_dir),
        'outputs': {
            'gt_overlay': str(overlay_path),
            'gt_bev': str(bev_path),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run_dataset_cli(dataset_name, argv=None):
    parser = argparse.ArgumentParser(
        description=(
            f'Visualize one {dataset_name} GT label in the camera image and '
            'ground-plane BEV.'))
    parser.add_argument('image', help='Path to one source camera image.')
    parser.add_argument(
        '--output-root',
        default=str(DEFAULT_OUTPUT_ROOT),
        help='Output root; files are grouped by dataset/scene/sample.',
    )
    args = parser.parse_args(argv)
    visualize_gt(args.image, dataset_name, args.output_root)
    return 0
