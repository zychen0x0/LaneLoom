#!/usr/bin/env python3
"""Visualize one OpenLane or ONCE image sample and its matching LiDAR."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mmengine.config import Config  # noqa: E402

from LaneLoom.datasets.openlane.dataset import (  # noqa: E402
    OpenLaneDataset,
    WAYMO_VEHICLE_TO_GROUND_AXES,
)
from LaneLoom.datasets.once.dataset import CAMERA_TO_GROUND  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
OPENLANE_CONFIG = REPO_ROOT / 'configs/laneloom_openlane_image_lidar.py'
ONCE_CONFIG = REPO_ROOT / 'configs/laneloom_once_image_lidar.py'
LANE_COLORS = (
    (40, 255, 80),
    (255, 190, 40),
    (60, 190, 255),
    (255, 90, 180),
    (180, 110, 255),
)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_configs():
    return {
        'openlane': Config.fromfile(OPENLANE_CONFIG),
        'once': Config.fromfile(ONCE_CONFIG),
    }


def _image_root(dataset_name, cfg):
    return Path(cfg.dataset_cfg.data_image_dir).expanduser().resolve()


def _detect_dataset(image_path, requested, configs):
    if requested != 'auto':
        return requested
    matches = [
        name for name, cfg in configs.items()
        if image_path.is_relative_to(_image_root(name, cfg))
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        'Cannot uniquely detect the dataset from the image path. Pass '
        '--dataset openlane or --dataset once, and set LANELOOM_DATA_ROOT '
        'when the data is outside the configured root.')


def _resolve_label(image_path, dataset_name, cfg):
    image_root = _image_root(dataset_name, cfg)
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(
            f'{dataset_name} image must be under {image_root}; set '
            'LANELOOM_DATA_ROOT to the parent data directory.') from error

    label_root = Path(cfg.dataset_cfg.data_label_dir).expanduser().resolve()
    if dataset_name == 'openlane':
        label_path = (label_root / relative).with_suffix('.json')
        if not label_path.is_file():
            raise FileNotFoundError(
                f'No OpenLane label found for image: {label_path}')
        return label_path, relative.parent.name, image_path.stem[:12]

    if len(relative.parts) < 3 or relative.parts[-2] != 'cam01':
        raise ValueError(
            'ONCE image path must end with <scene>/cam01/<frame>.<ext>.')
    scene_name = relative.parts[-3]
    frame_key = image_path.stem
    candidates = [
        label_root / split / scene_name / 'cam01' / f'{frame_key}.json'
        for split in ('train', 'val')
    ]
    label_path = next((path for path in candidates if path.is_file()), None)
    if label_path is None:
        raise FileNotFoundError(
            'No public ONCE train/val GT label found for image; checked: '
            + ', '.join(str(path) for path in candidates))
    return label_path, scene_name, frame_key


def _project_gt_lanes(label, dataset_name, cfg):
    if dataset_name == 'openlane':
        parser = OpenLaneDataset.__new__(OpenLaneDataset)
        parser.anchor_y_steps = np.asarray(
            cfg.anchor_y_steps, dtype=np.float32)
        parser.smooth_3d_gt = bool(cfg.smooth_3d_gt)
        parser.gt_outlier_residual = float(cfg.gt_outlier_residual)
        parser.max_lanes = int(cfg.max_lanes)
        parser.num_category = int(cfg.num_category)
        parser.org_h = parser.resize_h = int(cfg.org_h)
        parser.org_w = parser.resize_w = int(cfg.org_w)
        packed = parser._parse_openlane_gt(label)
        projection = parser._build_calib(label)['ground_to_raw_img']
        lanes = [
            (lane.numpy(), visibility.numpy().astype(bool))
            for lane, visibility in zip(
                packed['gt_lanes_3d'], packed['gt_lanes_visibility'])
            if visibility.any()
        ]
    else:
        projection = np.asarray(
            label['calibration'], dtype=np.float32).reshape(3, 4)
        lanes = []
        for lane in label.get('lanes', []):
            xyz = np.asarray(lane, dtype=np.float32)
            if xyz.ndim == 2 and xyz.shape[1] == 3:
                lanes.append((xyz, np.ones(xyz.shape[0], dtype=bool)))

    projected_lanes = []
    for xyz, visibility in lanes:
        if projection.shape[1] == 4:
            xyz = np.concatenate(
                [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        projected = (projection @ xyz.T).T
        depth = projected[:, 2]
        uv = np.full((xyz.shape[0], 2), np.nan, dtype=np.float32)
        positive = depth > 1e-5
        uv[positive] = projected[positive, :2] / depth[positive, None]
        projected_lanes.append((uv, visibility & positive))
    return projected_lanes


def _draw_gt_overlay(image, projected_lanes):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    for lane_index, (uv, visible) in enumerate(projected_lanes):
        color = LANE_COLORS[lane_index % len(LANE_COLORS)]
        previous = None
        for point, is_visible in zip(uv, visible):
            inside = (
                is_visible
                and np.isfinite(point).all()
                and 0 <= point[0] < width
                and 0 <= point[1] < height
            )
            if not inside:
                previous = None
                continue
            current = tuple(np.rint(point).astype(int))
            if previous is not None:
                draw.line((previous, current), fill=color, width=4)
            draw.ellipse(
                (current[0] - 3, current[1] - 3,
                 current[0] + 3, current[1] + 3),
                fill=color)
            previous = current
    return overlay


def _network_image_size(dataset_name, cfg):
    if dataset_name == 'once':
        return int(cfg.resize_w), int(cfg.resize_h)
    short_edge, long_edge = cfg.img_scale
    return int(long_edge), int(short_edge)


def _make_augmentation_grid(image, dataset_name, cfg, recipes, seed):
    image = image.resize(
        _network_image_size(dataset_name, cfg), Image.BILINEAR)
    base_array = np.asarray(image, dtype=np.uint8)
    builder = OpenLaneDataset.__new__(OpenLaneDataset)
    panels = []
    for recipe_index, recipe in enumerate(recipes):
        transform = builder._build_albu_compose(recipe.get('transforms', []))
        augmented = base_array.copy()
        if transform is not None:
            augmented = builder._apply_transform_with_seed(
                transform, augmented, seed + recipe_index)
        panels.append(Image.fromarray(augmented))

    columns = 3
    rows = (len(panels) + columns - 1) // columns
    panel_width, panel_height = panels[0].size
    canvas = Image.new(
        'RGB', (columns * panel_width, rows * panel_height), (255, 255, 255))
    for index, panel in enumerate(panels):
        x0 = index % columns * panel_width
        y0 = index // columns * panel_height
        canvas.paste(panel, (x0, y0))
    return canvas


def _load_openlane_points(cfg, scene_name, frame_key):
    lidar_root = Path(
        cfg.dataset_cfg.lidar_processed_dir).expanduser().resolve()
    scene_dir = lidar_root / scene_name
    info_path = scene_dir / f'{scene_name}.pkl'
    if not info_path.is_file():
        raise FileNotFoundError(
            f'OpenLane/Waymo LiDAR info is missing: {info_path}')
    with open(info_path, 'rb') as handle:
        frame_infos = pickle.load(handle)
    frame_info = next((
        info for info in frame_infos
        if str(info.get('metadata', {}).get('timestamp_micros', ''))[:12]
        == frame_key
    ), None)
    if frame_info is None:
        raise KeyError(f'No LiDAR frame matches {scene_name}/{frame_key}.')
    sample_index = int(frame_info['point_cloud']['sample_idx'])
    points_path = scene_dir / f'{sample_index:04d}.npy'
    points = np.load(points_path)
    if points.ndim != 2 or points.shape[1] < 5:
        raise ValueError(
            f'Unexpected OpenLane point shape {points.shape}: {points_path}')
    if points.shape[1] >= 6:
        valid = points[:, 5] <= 0
        if np.any(valid):
            points = points[valid]
    xyz = (WAYMO_VEHICLE_TO_GROUND_AXES @ points[:, :3].T).T
    return xyz.astype(np.float32), points[:, 3:5].astype(np.float32)


def _load_once_points(cfg, scene_name, frame_key):
    info_root = Path(cfg.dataset_cfg.info_dir).expanduser().resolve()
    lidar_root = Path(
        cfg.dataset_cfg.lidar_processed_dir).expanduser().resolve()
    info_path = info_root / scene_name / f'{scene_name}.json'
    sequence_info = _load_json(info_path)
    if not any(str(frame.get('frame_id')) == frame_key
               for frame in sequence_info.get('frames', [])):
        raise KeyError(
            f'No ONCE LiDAR frame matches {scene_name}/{frame_key}.')
    points_path = lidar_root / scene_name / 'lidar_roof' / f'{frame_key}.bin'
    raw = np.fromfile(points_path, dtype=np.float32)
    if raw.size == 0 or raw.size % 4:
        raise ValueError(f'Unexpected ONCE point data: {points_path}')
    points = raw.reshape(-1, 4)
    cam_to_velo = np.asarray(
        sequence_info['calib']['cam01']['cam_to_velo'],
        dtype=np.float32).reshape(4, 4)
    lidar_to_ground = CAMERA_TO_GROUND @ np.linalg.inv(cam_to_velo)
    xyz_h = np.concatenate([
        points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32)
    ], axis=1)
    xyz = (lidar_to_ground @ xyz_h.T).T[:, :3]
    features = np.column_stack((
        points[:, 3], np.zeros(points.shape[0], dtype=np.float32)))
    return xyz.astype(np.float32), features.astype(np.float32)


def _write_binary_ply(path, xyz, features):
    intensity = features[:, 0]
    low, high = np.percentile(intensity, [1, 99])
    scale = max(float(high - low), 1e-6)
    gray = np.clip((intensity - low) / scale * 255, 0, 255).astype(np.uint8)
    vertices = np.empty(xyz.shape[0], dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<f4'), ('elongation', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'], vertices['y'], vertices['z'] = xyz.T
    vertices['intensity'] = features[:, 0]
    vertices['elongation'] = features[:, 1]
    vertices['red'] = gray
    vertices['green'] = gray
    vertices['blue'] = gray
    header = '\n'.join([
        'ply',
        'format binary_little_endian 1.0',
        f'element vertex {len(vertices)}',
        'property float x',
        'property float y',
        'property float z',
        'property float intensity',
        'property float elongation',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'end_header',
        '',
    ]).encode('ascii')
    with open(path, 'wb') as handle:
        handle.write(header)
        vertices.tofile(handle)


def visualize_sample(
        image_path, dataset='auto', output_root=DEFAULT_OUTPUT_ROOT,
        seed=20260816):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f'Input image does not exist: {image_path}')
    configs = _load_configs()
    dataset_name = _detect_dataset(image_path, dataset, configs)
    cfg = configs[dataset_name]
    label_path, scene_name, frame_key = _resolve_label(
        image_path, dataset_name, cfg)
    output_dir = (
        Path(output_root).expanduser().resolve()
        / dataset_name / scene_name / frame_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert('RGB')
    label = _load_json(label_path)
    original_path = output_dir / '01_original.jpg'
    overlay_path = output_dir / '02_gt_overlay.jpg'
    augmentations_path = output_dir / '03_augmentations.jpg'
    points_path = output_dir / '04_points.ply'
    shutil.copyfile(image_path, original_path)
    _draw_gt_overlay(image, _project_gt_lanes(label, dataset_name, cfg)).save(
        overlay_path, quality=95)
    _make_augmentation_grid(
        image,
        dataset_name,
        cfg,
        configs['openlane'].image_augmentation_recipes,
        int(seed),
    ).save(augmentations_path, quality=95)

    if dataset_name == 'openlane':
        xyz, features = _load_openlane_points(cfg, scene_name, frame_key)
    else:
        xyz, features = _load_once_points(cfg, scene_name, frame_key)
    _write_binary_ply(points_path, xyz, features)

    report = {
        'dataset': dataset_name,
        'source_image': str(image_path),
        'source_label': str(label_path),
        'point_count': int(xyz.shape[0]),
        'output_dir': str(output_dir),
        'outputs': {
            'original': str(original_path),
            'gt_overlay': str(overlay_path),
            'augmentations': str(augmentations_path),
            'point_cloud': str(points_path),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Save the source image, 3D-GT overlay, named augmentations, and '
            'matching point cloud for one OpenLane or ONCE sample.'))
    parser.add_argument('image', help='Path to one source camera image.')
    parser.add_argument(
        '--dataset', choices=('auto', 'openlane', 'once'), default='auto',
        help='Dataset type. By default it is inferred from configured roots.')
    parser.add_argument(
        '--output-root', default=str(DEFAULT_OUTPUT_ROOT),
        help='Output root; files are grouped by dataset/scene/frame.')
    parser.add_argument(
        '--seed', type=int, default=20260816,
        help='Base random seed for reproducible named augmentations.')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    visualize_sample(
        args.image,
        dataset=args.dataset,
        output_root=args.output_root,
        seed=args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
