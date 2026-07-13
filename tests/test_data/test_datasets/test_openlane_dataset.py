from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from torch.utils.data import DataLoader

from LaneFormer.datasets import openlane_collate


def _load_cfg():
    config_path = (
        Path(__file__).resolve().parents[3]
        / 'configs'
        / 'laneformer_openlane_image_lidar.py'
    )
    return Config.fromfile(str(config_path))


def _require_openlane_data(cfg):
    required_paths = [
        Path(cfg.dataset_cfg.data_label_dir),
        Path(cfg.dataset_cfg.data_image_dir),
        Path(cfg.dataset_cfg.lidar_processed_dir),
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        pytest.skip('OpenLane/Waymo data is not available: {}'.format(missing_paths))


def _build_train_dataset(max_samples=1, dataset_cfg_overrides=None):
    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = max_samples
    if dataset_cfg_overrides:
        dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
        dataset_cfg['dataset_cfg'].update(dataset_cfg_overrides)
    return DATASETS.build(dataset_cfg), cfg


def _build_test_dataset(max_samples=1):
    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.test_dataloader['dataset'])
    dataset_cfg['max_samples'] = max_samples
    return DATASETS.build(dataset_cfg), cfg


def _assert_4x4_matrices(matrices, num_views):
    assert isinstance(matrices, list)
    assert len(matrices) == num_views
    for matrix in matrices:
        assert np.asarray(matrix).shape == (4, 4)


def _lidar_feat_dim(cfg):
    return cfg.use_dim if isinstance(cfg.use_dim, int) else len(cfg.use_dim)


def _queue_as_list(queue):
    if isinstance(queue, dict):
        return [queue[idx] for idx in sorted(queue.keys())]
    return list(queue)


def _assert_temporal_frame_meta(frame_meta, num_views):
    standard_keys = [
        'sample_idx',
        'pts_filename',
        'sweeps',
        'timestamp',
        'timestamp_micros',
        'ego2global_translation',
        'ego2global_rotation',
        'lidar2ego_translation',
        'lidar2ego_rotation',
        'prev',
        'next',
        'scene_token',
        'frame_idx',
        'img_filename',
        'lidar2img',
        'cam2img',
        'lidar2cam',
        'caminfo',
        'ori_shape',
        'img_shape',
        'pad_shape',
        'img_norm_cfg',
        'img_aug_matrix',
        'lidaradj2lidarcurr',
        'lane2img',
        'openlane_calib',
    ]
    for key in standard_keys:
        assert key in frame_meta

    assert isinstance(frame_meta['timestamp'], float)
    assert isinstance(frame_meta['timestamp_micros'], int)
    assert frame_meta['ego2global_translation'].shape == (3,)
    assert frame_meta['ego2global_rotation'].shape == (3, 3)
    assert frame_meta['lidar2ego_translation'].shape == (3,)
    assert frame_meta['lidar2ego_rotation'].shape == (3, 3)
    assert frame_meta['lidaradj2lidarcurr'].shape == (4, 4)
    assert frame_meta['img_aug_matrix'].shape == (3, 3)
    _assert_4x4_matrices(frame_meta['lidar2img'], num_views)
    _assert_4x4_matrices(frame_meta['cam2img'], num_views)
    _assert_4x4_matrices(frame_meta['lidar2cam'], num_views)
    _assert_4x4_matrices(frame_meta['lane2img'], num_views)
    assert len(frame_meta['caminfo']) == num_views
    assert frame_meta['caminfo'][0]['cam_intrinsic'].shape == (3, 3)
    assert 'cam_to_vehicle' in frame_meta['openlane_calib']


def test_openlane_getitem_loads_multimodal_temporal_sample():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(max_samples=1)
    data = dataset[0]
    len_queue = len(cfg.frames)
    img_h, img_w = dataset.resize_h, dataset.resize_w

    # img: [T, num_views, C, H, W]. num_views 是同一前视图像的多增强视图数量。
    assert data['img'].shape == (len_queue, cfg.num_views, 3, img_h, img_w)

    # lidar: 变长点云，最后一维是 queue_index。
    assert isinstance(data['lidar'], torch.Tensor)
    assert data['lidar'].ndim == 2
    assert data['lidar'].shape[1] == _lidar_feat_dim(cfg)
    if data['lidar'].numel() > 0:
        assert data['lidar'][:, -1].min() >= 0
        assert data['lidar'][:, -1].max() < len_queue

    # OpenLane training GT tensors consumed by lane heads.
    assert data['gt_lanes_3d'].shape == (cfg.max_lanes, cfg.num_y_steps, 3)
    assert data['gt_lanes_uv'].shape == (cfg.max_lanes, cfg.num_y_steps, 2)
    assert data['gt_lanes_visibility'].shape == (cfg.max_lanes, cfg.num_y_steps)
    assert data['gt_lanes_category'].shape == (cfg.max_lanes,)
    if cfg.temporal_gt_augmentation:
        assert data['gt_lanes_category_enhanced'].shape == (cfg.max_lanes,)
    assert data['ground_lanes'].shape[0] == cfg.max_lanes
    assert dataset.metainfo['dataset_name'] == 'OpenLane'
    assert dataset.metainfo['classes'] == tuple(cfg.class_names)
    assert dataset.metainfo['num_category'] == cfg.num_category

    img_metas = data['img_metas']
    assert img_metas['num_history_frames'] == len_queue - 1
    assert img_metas['sampling_mode'] in {'distance', 'neighbor'}
    assert len(img_metas['frame_keys']) == len_queue
    assert len(img_metas['queue']) == len_queue
    assert img_metas['num_views'] == cfg.num_views
    assert img_metas['view_mode'] == cfg.view_mode
    assert img_metas['view_mask'].shape == (len_queue, cfg.num_views)
    assert img_metas['view_mask'].dtype == torch.bool
    assert torch.all(img_metas['view_mask'])

    queue = _queue_as_list(img_metas['queue'])
    for frame_meta in queue:
        _assert_temporal_frame_meta(frame_meta, cfg.num_views)

    current_meta = img_metas['current']
    _assert_temporal_frame_meta(current_meta, cfg.num_views)
    assert np.allclose(
        queue[-1]['lidaradj2lidarcurr'],
        np.eye(4, dtype=np.float32),
        atol=1e-4)

    # OpenLane-specific calibration remains available, but not under confusing top-level names.
    for legacy_key in ('calib', 'cam_to_vehicle', 'vehicle_to_img', 'ground_to_image'):
        assert legacy_key not in current_meta


def test_openlane_first_segment_frame_without_history_is_padded():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(max_samples=128)

    for idx, label_path in enumerate(dataset.label_files):
        scene_name = label_path.parent.name
        current_key = label_path.stem[:12]
        label_scene = dataset._load_label_scene(scene_name)
        lidar_scene = dataset._load_lidar_scene(scene_name)
        label_is_first = label_scene['keys'][0] == current_key
        lidar_has_no_earlier_frame = (
            lidar_scene is None
            or current_key not in lidar_scene['frames']
            or lidar_scene['keys'].index(current_key) == 0
        )
        if label_is_first and lidar_has_no_earlier_frame:
            data = dataset[idx]
            frame_keys = data['img_metas']['frame_keys']
            len_queue = len(cfg.frames)

            assert len(frame_keys) == len_queue
            assert frame_keys[-1] == current_key
            assert frame_keys == [current_key] * len_queue
            assert data['img'].shape[0] == len_queue
            assert data['img_metas']['queue'][len_queue - 1]['lidaradj2lidarcurr'].shape == (4, 4)
            for queue_idx in range(len_queue):
                transform = data['img_metas']['queue'][queue_idx]['lidaradj2lidarcurr']
                assert np.allclose(transform, np.eye(4, dtype=np.float32), atol=1e-4)
            break
    else:
        pytest.skip('No segment-first OpenLane sample without earlier LiDAR frame found.')


def test_openlane_dataloader_collates_points_as_list():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(max_samples=1)
    img_h, img_w = dataset.resize_h, dataset.resize_w
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_dataloader['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=openlane_collate)
    batch = next(iter(dataloader))

    assert batch['img'].shape == (1, len(cfg.frames), cfg.num_views, 3, img_h, img_w)
    assert batch['view_mask'].shape == (1, len(cfg.frames), cfg.num_views)
    assert batch['view_mask'].dtype == torch.bool
    assert isinstance(batch['lidar'], list)
    assert len(batch['lidar']) == 1
    assert batch['lidar'][0].ndim == 2
    assert batch['lidar'][0].shape[1] == _lidar_feat_dim(cfg)
    assert batch['gt_lanes_3d'].shape == (1, cfg.max_lanes, cfg.num_y_steps, 3)


def test_openlane_get_data_info_exposes_temporal_calibration_keys():
    dataset, cfg = _build_train_dataset(max_samples=1)
    data_info = dataset.get_data_info(0)
    img_h, img_w = dataset.resize_h, dataset.resize_w

    assert data_info['img_scale'] == cfg.img_scale
    assert data_info['img_shape'] == (img_h, img_w, 3)
    assert data_info['pad_shape'] == (img_h, img_w, 3)

    standard_keys = [
        'sample_idx',
        'pts_filename',
        'sweeps',
        'ego2global_translation',
        'ego2global_rotation',
        'lidar2ego_translation',
        'lidar2ego_rotation',
        'timestamp',
        'img_filename',
        'lidar2img',
        'cam2img',
        'lidar2cam',
    ]
    for key in standard_keys:
        assert key in data_info
    assert 'ann_info' in data_info


def test_openlane_config_uses_mmengine_style_fields():
    cfg = _load_cfg()

    assert 'train_dataloader' in cfg
    assert 'val_dataloader' in cfg
    assert 'test_dataloader' in cfg
    assert 'optim_wrapper' in cfg
    assert 'param_scheduler' in cfg
    assert 'train_cfg' in cfg

    for legacy_key in ('data', 'optimizer_config', 'lr_config', 'runner'):
        assert legacy_key not in cfg

    assert 'file_client_args' not in cfg.dataset_cfg


def test_openlane_test_dataloader_uses_recursive_official_test_split():
    dataset, _ = _build_test_dataset(max_samples=1)

    assert dataset.split == 'test'
    assert dataset.label_files
    assert 'test' in dataset.label_files[0].parts
    # Official test labels are grouped as test/<scenario>/<segment>/<frame>.
    assert len(dataset.label_files[0].relative_to(dataset.label_root).parts) >= 4
    assert dataset._scenario_name(dataset.label_files[0]).endswith('_case')


def test_openlane_original_mode_repeats_raw_views():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(
        max_samples=1,
        dataset_cfg_overrides=dict(
            num_views=6,
            view_mode='original',
        ))
    data = dataset[0]

    assert data['img_metas']['view_mask'].shape == (len(cfg.frames), dataset.num_views)
    assert torch.all(data['img_metas']['view_mask'])
    for view_idx in range(1, dataset.num_views):
        assert torch.allclose(data['img'][:, 0], data['img'][:, view_idx])


def test_openlane_load_interval_uses_d5_subset():
    full_dataset, _ = _build_train_dataset(
        max_samples=20, dataset_cfg_overrides=dict(load_interval=1))
    d5_dataset, _ = _build_train_dataset(
        max_samples=20,
        dataset_cfg_overrides=dict(load_interval=5))

    assert len(full_dataset) == 20
    assert len(d5_dataset) == 4
    assert d5_dataset.label_files == full_dataset.label_files[::5]


def test_openlane_history_frames_have_relative_transform_when_available():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(max_samples=32)
    for idx in range(len(dataset)):
        data = dataset[idx]
        frame_keys = data['img_metas']['frame_keys']
        if len(set(frame_keys)) <= 1:
            continue

        queue = data['img_metas']['queue']
        current_transform = queue[len(cfg.frames) - 1]['lidaradj2lidarcurr']
        assert np.allclose(current_transform, np.eye(4, dtype=np.float32), atol=1e-4)

        history_transforms = [
            queue[queue_idx]['lidaradj2lidarcurr']
            for queue_idx in range(len(cfg.frames) - 1)
        ]
        assert any(
            not np.allclose(transform, np.eye(4, dtype=np.float32), atol=1e-4)
            for transform in history_transforms
        )
        break
    else:
        pytest.skip('No non-duplicated history frames found in the sampled OpenLane subset.')
