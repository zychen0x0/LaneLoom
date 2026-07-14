"""Inspect GroundGrid extraction and road curvature on an OpenLane sample."""

import argparse
import json
import shutil
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--index', type=int, default=None)
    selection.add_argument(
        '--image-path', default=None,
        help='Select an OpenLane sample directly from its original image.')
    selection.add_argument(
        '--lidar-file', default=None,
        help='Optional .npy point cloud; bypasses the OpenLane dataset loader.')
    parser.add_argument(
        '--lidar-frame', choices=('waymo_vehicle', 'openlane_ground'),
        default='waymo_vehicle',
        help='Coordinate frame of --lidar-file (default: raw Waymo vehicle).')
    parser.add_argument(
        '--output-prefix', default=None,
        help=('Write <prefix>.png, <prefix>_image.<ext>, and '
              '<prefix>_lidar_current_ground.ply.'))
    return parser.parse_args()


def resolve_image_sample(cfg, image_path):
    """Resolve an original OpenLane image to its label and loader split."""
    image_path = Path(image_path).expanduser().resolve()
    image_root = Path(cfg.dataset_cfg.data_image_dir).expanduser().resolve()
    label_root = Path(cfg.dataset_cfg.data_label_dir).expanduser().resolve()
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(
            f'image must be under configured image root: {image_root}') from error
    if len(relative.parts) < 3:
        raise ValueError(
            'image path must include split/segment/frame below image root.')
    split_directory = relative.parts[0]
    split_names = {
        str(cfg.dataset_cfg.train_split): 'train',
        str(cfg.dataset_cfg.val_split): 'val',
        str(cfg.dataset_cfg.test_split): 'test',
    }
    if split_directory not in split_names:
        raise ValueError(f'unknown OpenLane split directory: {split_directory}')
    label_path = (label_root / relative).with_suffix('.json')
    if not label_path.is_file():
        matches = [
            path for path in (label_root / split_directory).rglob(
                f'{image_path.stem}.json')
            if image_path.parent.name in path.parts
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f'expected one label for {image_path}, found {len(matches)}')
        label_path = matches[0]
    return label_path.resolve(), split_names[split_directory]


def _dataset_source(sample, split, index=None):
    meta = sample['img_metas']
    current = meta.get('current', {})
    image_paths = current.get('img_filename', ())
    return {
        'source': 'OpenLaneDataset',
        'split': split,
        'index': index,
        'scene_name': meta.get('scene_name', ''),
        'frame_key': meta.get('current_key', ''),
        'frame_keys': meta.get('frame_keys', []),
        'image_path': str(image_paths[0]) if image_paths else None,
        'lidar_path': current.get('pts_filename'),
        'lidar_frame': 'waymo_vehicle',
    }


def load_points(cfg, args):
    if args.lidar_file:
        array = _load_lidar_points(args.lidar_file, args.lidar_frame)
        return torch.from_numpy(array), {
            'source': str(Path(args.lidar_file).resolve()),
            'input_frame': args.lidar_frame,
            'analysis_frame': 'openlane_ground',
            'lidar_path': str(Path(args.lidar_file).resolve()),
            'lidar_frame': args.lidar_frame}

    if args.image_path:
        label_path, split = resolve_image_sample(cfg, args.image_path)
        loader_cfg = cfg[f'{split}_dataloader']
        dataset_cfg = dict(loader_cfg['dataset'])
        dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
        dataset_cfg['dataset_cfg']['load_interval'] = 1
        dataset_cfg['max_samples'] = 1
        dataset = DATASETS.build(dataset_cfg)
        dataset.label_files = [label_path]
        sample = dataset[0]
        return sample['lidar'], _dataset_source(sample, split)

    index = 0 if args.index is None else args.index
    loader_cfg = cfg[f'{args.split}_dataloader']
    dataset_cfg = dict(loader_cfg['dataset'])
    load_interval = int(dataset_cfg.get('dataset_cfg', {}).get(
        'load_interval', 1))
    dataset_cfg['max_samples'] = (index + 1) * load_interval
    dataset = DATASETS.build(dataset_cfg)
    if index >= len(dataset):
        raise IndexError(
            f'index {index} is outside dataset length {len(dataset)}')
    sample = dataset[index]
    return sample['lidar'], _dataset_source(sample, args.split, index=index)


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


def _load_lidar_points(path, coordinate_frame):
    array = np.load(Path(path).expanduser().resolve())
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError('LiDAR file must have shape [N, >=3].')
    if array.shape[1] >= 6:
        valid_return = array[:, 5] <= 0
        if np.any(valid_return):
            array = array[valid_return]
    array = array.astype(np.float32)
    if coordinate_frame == 'waymo_vehicle':
        array[:, :3] = (
            WAYMO_VEHICLE_TO_GROUND_AXES @ array[:, :3].T).T
    return array


def _write_xyz_ply(path, xyz, source_path=None):
    with Path(path).open('w', encoding='utf-8') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        if source_path is not None:
            handle.write(f'comment source {Path(source_path).resolve()}\n')
        handle.write('comment content current_frame_unsegmented_valid_returns\n')
        handle.write(
            'comment coordinates openlane_ground_x_lateral_y_forward_z_up\n')
        handle.write(f'element vertex {xyz.shape[0]}\n')
        handle.write('property float x\nproperty float y\nproperty float z\n')
        handle.write('end_header\n')
        np.savetxt(handle, xyz[:, :3], fmt='%.6f')


def save_outputs(prefix, points, result, source):
    prefix = Path(prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    valid = result['road_valid'][0]

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
    if valid.any():
        axes[1].legend(loc='best')
    scene_name = source.get('scene_name')
    frame_key = source.get('frame_key')
    if scene_name or frame_key:
        figure.suptitle(f'{scene_name} / {frame_key}'.strip(' /'))
    figure.tight_layout()
    visualization_path = prefix.with_suffix('.png')
    figure.savefig(visualization_path, dpi=160)
    plt.close(figure)

    outputs = {'visualization': visualization_path}
    image_path = source.get('image_path')
    if image_path:
        image_path = Path(image_path).expanduser().resolve()
        copied_image = prefix.parent / f'{prefix.name}_image{image_path.suffix}'
        shutil.copy2(image_path, copied_image)
        outputs['original_image'] = copied_image
    lidar_path = source.get('lidar_path')
    if lidar_path:
        lidar_points = _load_lidar_points(
            lidar_path, source.get('lidar_frame', 'waymo_vehicle'))
    else:
        lidar_points = point_array
    lidar_output = prefix.parent / f'{prefix.name}_lidar_current_ground.ply'
    _write_xyz_ply(lidar_output, lidar_points, source_path=lidar_path)
    outputs['original_lidar_current_ground'] = lidar_output
    return outputs


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
        outputs = save_outputs(args.output_prefix, points, result, source)
        print('Outputs:')
        for name, path in outputs.items():
            print(f'  {name}: {path}')


if __name__ == '__main__':
    main()
