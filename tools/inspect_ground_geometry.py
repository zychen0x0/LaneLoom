"""Inspect GroundGrid extraction and road curvature on an OpenLane sample."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS, MODELS

import LaneFormer  # noqa: F401 - register project components.
from LaneFormer.datasets.openlane_dataset import WAYMO_VEHICLE_TO_GROUND_AXES


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract ground points and road geometry from LiDAR.')
    parser.add_argument(
        'config', help='LaneFormer config containing ground_prior_extractor.')
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        default='train')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument(
        '--lidar-file', default=None,
        help='Optional .npy point cloud; bypasses the OpenLane dataset loader.')
    parser.add_argument(
        '--lidar-frame', choices=('waymo_vehicle', 'openlane_ground'),
        default='waymo_vehicle',
        help='Coordinate frame of --lidar-file (default: raw Waymo vehicle).')
    parser.add_argument(
        '--output-prefix', default=None,
        help='Write <prefix>.json, <prefix>.npz and <prefix>.png.')
    return parser.parse_args()


def load_points(cfg, args):
    if args.lidar_file:
        array = np.load(Path(args.lidar_file).expanduser().resolve())
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError('LiDAR file must have shape [N, >=3].')
        if array.shape[1] >= 6:
            valid_return = array[:, 5] <= 0
            if np.any(valid_return):
                array = array[valid_return]
        array = array.astype(np.float32)
        if args.lidar_frame == 'waymo_vehicle':
            array[:, :3] = (
                WAYMO_VEHICLE_TO_GROUND_AXES @ array[:, :3].T).T
        return torch.from_numpy(array.astype(np.float32)), {
            'source': str(Path(args.lidar_file).resolve()),
            'input_frame': args.lidar_frame,
            'analysis_frame': 'openlane_ground'}

    loader_cfg = cfg[f'{args.split}_dataloader']
    dataset_cfg = dict(loader_cfg['dataset'])
    load_interval = int(dataset_cfg.get('dataset_cfg', {}).get(
        'load_interval', 1))
    dataset_cfg['max_samples'] = (args.index + 1) * load_interval
    dataset = DATASETS.build(dataset_cfg)
    if args.index >= len(dataset):
        raise IndexError(
            f'index {args.index} is outside dataset length {len(dataset)}')
    sample = dataset[args.index]
    meta = sample['img_metas']
    return sample['lidar'], {
        'source': 'OpenLaneDataset',
        'split': args.split,
        'index': args.index,
        'scene_name': meta.get('scene_name', ''),
        'frame_key': meta.get('current_key', ''),
        'frame_keys': meta.get('frame_keys', []),
    }


def summarize(points, result, source):
    valid = result['road_valid'][0]
    horizontal = result['horizontal_curvature'][0][valid]
    vertical = result['vertical_curvature'][0][valid]

    def per_road(values, reduction):
        return [float(reduction(road).cpu()) for road in values]

    summary = dict(source)
    summary.update(
        input_points=int(points.size(0)),
        ground_points=int(result['ground_masks'][0].sum().cpu()),
        ground_ratio=float(result['ground_masks'][0].float().mean().cpu()),
        valid_road_hypotheses=int(valid.sum().cpu()),
        road_direction_degrees=(
            result['road_directions'][0][valid].rad2deg().cpu().tolist()),
        median_horizontal_curvature=per_road(horizontal, torch.median),
        max_abs_horizontal_curvature=per_road(horizontal.abs(), torch.max),
        median_vertical_curvature=per_road(vertical, torch.median),
        max_abs_vertical_curvature=per_road(vertical.abs(), torch.max),
    )
    return summary


def save_outputs(prefix, points, result, summary):
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    valid = result['road_valid'][0]
    np.savez_compressed(
        prefix.with_suffix('.npz'),
        points=points.cpu().numpy(),
        ground_mask=result['ground_masks'][0].cpu().numpy(),
        road_controls=result['road_controls'][0][valid].cpu().numpy(),
        road_samples=result['road_samples'][0][valid].cpu().numpy(),
        horizontal_curvature=(
            result['horizontal_curvature'][0][valid].cpu().numpy()),
        vertical_curvature=(
            result['vertical_curvature'][0][valid].cpu().numpy()))

    point_array = points.cpu().numpy()
    ground_mask = result['ground_masks'][0].cpu().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    stride = max(1, point_array.shape[0] // 50000)
    axes[0].scatter(point_array[::stride, 0], point_array[::stride, 1],
                    s=.2, c='0.75', label='all points')
    ground = point_array[ground_mask]
    ground_stride = max(1, ground.shape[0] // 50000)
    axes[0].scatter(ground[::ground_stride, 0], ground[::ground_stride, 1],
                    s=.3, c=ground[::ground_stride, 2], cmap='terrain',
                    label='ground')
    for road_index, road in enumerate(result['road_samples'][0][valid].cpu()):
        axes[0].plot(road[:, 0], road[:, 1], linewidth=2,
                     label=f'road {road_index}')
    axes[0].set(title='BEV ground and road hypotheses', xlabel='x lateral [m]',
                ylabel='y forward [m]', aspect='equal')
    axes[0].legend(loc='best')

    anchors = result['anchor_y_steps'].cpu().numpy()
    for road_index, road_valid_index in enumerate(torch.where(valid)[0]):
        axes[1].plot(
            anchors,
            result['horizontal_curvature'][0, road_valid_index].cpu().numpy(),
            label=f'road {road_index} horizontal')
        axes[1].plot(
            anchors,
            result['vertical_curvature'][0, road_valid_index].cpu().numpy(),
            linestyle='--', label=f'road {road_index} vertical')
    axes[1].axhline(0, color='0.5', linewidth=.5)
    axes[1].set(title='Road curvature priors', xlabel='y anchor [m]',
                ylabel='curvature [1/m]')
    axes[1].legend(loc='best')
    figure.tight_layout()
    figure.savefig(prefix.with_suffix('.png'), dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    cfg = Config.fromfile(str(Path(args.config).expanduser().resolve()))
    points, source = load_points(cfg, args)
    extractor = MODELS.build(cfg.model.ground_prior_extractor)
    with torch.no_grad():
        result = extractor([points])
    summary = summarize(points, result, source)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_prefix:
        save_outputs(args.output_prefix, points, result, summary)


if __name__ == '__main__':
    main()
