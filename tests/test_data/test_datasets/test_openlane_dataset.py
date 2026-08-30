from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from torch.utils.data import DataLoader

from LaneLoom.datasets import OpenLaneDataset, openlane_collate
from LaneLoom.datasets.openlane.dataset import (
    _build_cam_to_ground_lane,
    _vehicle_transform_to_ground,
)


def _load_cfg():
    config_path = (
        Path(__file__).resolve().parents[3]
        / 'configs'
        / 'main'
        / 'laneloom_openlane_image_lidar.py'
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


def _bare_gt_dataset(anchor_y_steps=None):
    """Build the small dataset surface needed by GT smoothing unit tests."""
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.anchor_y_steps = np.asarray(
        anchor_y_steps if anchor_y_steps is not None else np.arange(0, 101, 2),
        dtype=np.float32)
    dataset.smooth_3d_gt = True
    dataset.gt_outlier_residual = 0.75
    dataset.max_lanes = 4
    dataset.num_category = 21
    return dataset


def test_smooth_lane_reports_first_fit_status_for_accepted_full_fit():
    dataset = _bare_gt_dataset(np.arange(0, 51, 10))
    y = np.asarray([10., 20., 30., 40.], dtype=np.float32)
    raw = np.column_stack(
        (0.01 * y ** 2, y, 0.1 * y)).astype(np.float32)

    smoothed, keep, fit_status = dataset._smooth_lane_ground(raw)

    assert smoothed.shape == (4, 3)
    assert keep.all()
    assert fit_status == 'first_fit'


def test_smooth_lane_reports_robust_refit_status_when_outlier_is_removed():
    dataset = _bare_gt_dataset()
    y = np.arange(0., 101., 10., dtype=np.float32)
    z = 0.002 * y
    z[5] += 2.
    raw = np.column_stack((0.01 * y, y, z)).astype(np.float32)

    _, keep, fit_status = dataset._smooth_lane_ground(raw)

    assert keep.sum() == len(raw) - 1
    assert fit_status == 'robust_refit'


def test_smooth_lane_reports_linear_fallback_when_cubic_is_unavailable():
    dataset = _bare_gt_dataset()
    raw = np.asarray([
        [0., 20., 0.],
        [0.2, 30., 0.1],
        [0.4, 40., 0.2],
    ], dtype=np.float32)

    smoothed, keep, fit_status = dataset._smooth_lane_ground(raw)

    assert smoothed.shape[0] > 0
    assert keep.all()
    assert fit_status == 'linear_fallback'


def test_smooth_lane_reports_linear_fallback_when_smoothing_is_disabled():
    dataset = _bare_gt_dataset()
    dataset.smooth_3d_gt = False
    raw = np.asarray([
        [0., 20., 0.],
        [0.2, 30., 0.1],
        [0.4, 40., 0.2],
        [0.6, 50., 0.3],
    ], dtype=np.float32)

    unsmoothed, keep, fit_status = dataset._smooth_lane_ground(raw)

    assert np.array_equal(unsmoothed, raw)
    assert keep.all()
    assert fit_status == 'linear_fallback'


def test_smooth_lane_fits_full_raw_support_but_samples_only_model_intersection():
    dataset = _bare_gt_dataset(np.arange(0, 101, 10))
    dataset.gt_outlier_residual = 1e6
    y = np.asarray([20., 40., 60., 80., 100., 115., 130.])
    x = np.asarray([0., 0.4, 1.2, 2.8, 5.5, 9.0, 14.0])
    z = np.asarray([0., 0.1, 0.2, 0.5, 1.0, 1.8, 3.0])
    raw = np.column_stack((x, y, z)).astype(np.float32)

    smoothed, _, _ = dataset._smooth_lane_ground(raw)

    t = (y - y.min()) / (y.max() - y.min())
    design = np.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                       3 * (1 - t) * t ** 2, t ** 3), axis=1)
    controls = np.linalg.lstsq(design, raw[:, (0, 2)], rcond=None)[0]
    sample_y = np.arange(20., 101., 10.)
    sample_t = (sample_y - y.min()) / (y.max() - y.min())
    sample_design = np.stack(
        ((1 - sample_t) ** 3, 3 * (1 - sample_t) ** 2 * sample_t,
         3 * (1 - sample_t) * sample_t ** 2, sample_t ** 3), axis=1)

    assert np.array_equal(smoothed[:, 1], sample_y)
    assert np.allclose(smoothed[:, (0, 2)], sample_design @ controls, atol=1e-5)
    assert smoothed[:, 1].max() == 100.


def test_smooth_lane_ignores_gt_entirely_outside_model_y_range():
    dataset = _bare_gt_dataset()
    raw = np.asarray([
        [1., 110., 0.], [1.2, 115., 0.1], [1.5, 120., 0.2],
        [1.9, 125., 0.4], [2.4, 130., 0.7],
    ], dtype=np.float32)

    smoothed, keep, fit_status = dataset._smooth_lane_ground(raw)

    assert smoothed.shape == (0, 3)
    assert keep.all()  # The source annotation itself remains intact for audit/debug.
    assert fit_status == 'rejected'


def test_smooth_lane_uses_only_anchors_strictly_inside_raw_support():
    dataset = _bare_gt_dataset()
    y = np.asarray([20.1, 21.0, 22.0, 23.0, 24.1], dtype=np.float32)
    raw = np.column_stack((0.1 * y, y, 0.01 * y)).astype(np.float32)

    smoothed, _, _ = dataset._smooth_lane_ground(raw)

    assert np.array_equal(smoothed[:, 1], np.asarray([22., 24.], dtype=np.float32))


def test_smooth_lane_rejects_ill_conditioned_second_fit():
    dataset = _bare_gt_dataset()
    raw = np.asarray([
        [0.408259, 10.491498, -0.171423],
        [-1.563939, 10.617766, -0.205909],
        [-4.619934, 18.279676, -0.197012],
        [-6.894671, 18.759672, -0.202094],
        [-7.824201, 18.990381, -0.210537],
        [-7.672631, 19.098595, -0.183240],
        [-7.141119, 19.204458, -0.191206],
        [-6.964063, 19.310049, -0.176798],
        [-7.501328, 19.421003, -0.207724],
        [-8.335070, 19.631893, -0.218837],
        [-7.861882, 19.909403, -0.188786],
    ], dtype=np.float32)

    smoothed, keep, _ = dataset._smooth_lane_ground(raw)

    assert np.isfinite(smoothed).all()
    assert np.abs(smoothed[:, 0]).max() < 20.
    assert keep.all()


def test_pack_lane_targets_skips_out_of_range_lane_without_consuming_slot():
    dataset = _bare_gt_dataset()
    outside = np.asarray(
        [[1., 110., 0.], [2., 120., 0.]], dtype=np.float32)
    inside = np.asarray(
        [[2.01, 20.1, 0.2], [2.41, 24.1, 0.24]], dtype=np.float32)
    empty_uv = np.zeros((0, 2), dtype=np.float32)
    empty_uvy = np.zeros((0, 3), dtype=np.float32)

    packed = dataset._pack_lane_targets(
        [outside, inside], [empty_uv, empty_uv], [empty_uvy, empty_uvy],
        [3, 7], track_ids=[30, 70])

    expected_visibility = torch.zeros(len(dataset.anchor_y_steps))
    expected_visibility[[11, 12]] = 1.  # y=22m and y=24m only.
    assert packed['gt_lanes_category'][0].item() == 7
    assert packed['gt_lane_track_ids'][0].item() == 70
    assert torch.equal(packed['gt_lanes_visibility'][0], expected_visibility)
    assert packed['gt_lanes_visibility'][1:].sum().item() == 0


def test_temporal_gt_skips_out_of_range_history_record_without_global_geometry():
    dataset = _bare_gt_dataset()
    current_ground = np.asarray(
        [[0., 20., 0.], [0., 22., 0.]], dtype=np.float32)
    current_record = {
        'track_id': 11,
        'ground': current_ground,
        'global': current_ground.copy(),
        'category': 1,
    }
    outside_history_record = dict(
        track_id=11, ground=np.zeros((0, 3), dtype=np.float32), category=1)
    current_label = dict(frame='current')
    history_label = dict(frame='history')

    dataset._build_calib = lambda label: {}
    dataset._global_from_ground = lambda label, meta: np.eye(4, dtype=np.float32)
    dataset._parse_lane_records = (
        lambda label, meta=None, is_history=False:
        [current_record] if label is current_label else [outside_history_record])
    dataset._clip_current_roi_and_fov = lambda points, calib: points
    dataset._split_lane_segments = lambda points, max_gap: [points]
    dataset._arc_length_resample = lambda segments, step: segments

    num_y = len(dataset.anchor_y_steps)
    packed = dict(
        gt_lanes_3d=torch.zeros(dataset.max_lanes, num_y, 3),
        gt_lanes_visibility=torch.zeros(dataset.max_lanes, num_y),
        gt_lanes_category=torch.zeros(dataset.max_lanes, dtype=torch.long),
        gt_lane_track_ids=torch.tensor([11, -1, -1, -1]),
    )
    dataset._parse_openlane_gt = lambda label: packed

    enhanced = dataset._build_temporal_enhanced_gt(
        [history_label, current_label], [{}, {}])

    assert enhanced['temporal_gt_associations'] == []
    assert enhanced['gt_lane_track_ids_enhanced'][0].item() == 11


def test_temporal_gt_rejects_linear_fallback_history_only_lane():
    dataset = _bare_gt_dataset()
    y = np.arange(20., 31., 2., dtype=np.float32)
    trusted_ground = np.column_stack(
        (np.zeros_like(y), y, np.zeros_like(y)))
    history_only_ground = np.column_stack(
        (np.full_like(y, 3.), y, np.zeros_like(y)))
    current_record = {
        'track_id': 11, 'ground': trusted_ground,
        'global': trusted_ground.copy(), 'category': 1,
        'fit_status': 'first_fit',
    }
    trusted_history_record = {
        'track_id': 11, 'ground': trusted_ground,
        'global': trusted_ground.copy(), 'category': 1,
        'fit_status': 'first_fit',
    }
    fallback_history_record = {
        'track_id': 22, 'ground': history_only_ground,
        'global': history_only_ground.copy(), 'category': 1,
        'fit_status': 'linear_fallback',
    }
    current_label = dict(frame='current')
    history_label = dict(frame='history')

    dataset._build_calib = lambda label: {}
    dataset._global_from_ground = lambda label, meta: np.eye(4, dtype=np.float32)
    dataset._parse_lane_records = (
        lambda label, meta=None, is_history=False:
        [current_record] if label is current_label else
        [trusted_history_record, fallback_history_record])
    dataset._clip_current_roi_and_fov = lambda points, calib: points

    num_y = len(dataset.anchor_y_steps)
    current_lanes = torch.zeros(dataset.max_lanes, num_y, 3)
    current_visibility = torch.zeros(dataset.max_lanes, num_y)
    current_anchor_mask = np.isin(dataset.anchor_y_steps, y)
    current_lanes[0, current_anchor_mask, 1] = torch.from_numpy(y)
    current_visibility[0, current_anchor_mask] = 1.
    packed = dict(
        gt_lanes_3d=current_lanes,
        gt_lanes_visibility=current_visibility,
        gt_lanes_category=torch.tensor([1, 0, 0, 0]),
        gt_lane_track_ids=torch.tensor([11, -1, -1, -1]),
    )
    dataset._parse_openlane_gt = lambda label: packed

    enhanced = dataset._build_temporal_enhanced_gt(
        [history_label, current_label], [{}, {}])

    assert 22 not in enhanced['gt_lane_track_ids_enhanced'].tolist()
    assert any(
        association.get('reason') == 'history_only_linear_fallback'
        for association in enhanced['temporal_gt_associations'])


def test_temporal_gt_uses_strict_match_radius_with_relaxed_overlap_gates():
    dataset = _bare_gt_dataset()
    y_short = np.arange(20., 25., dtype=np.float32)
    y_long = np.arange(20., 26., dtype=np.float32)
    current_partial = np.column_stack(
        (np.zeros_like(y_short), y_short, np.zeros_like(y_short)))
    history_partial = np.column_stack((
        np.asarray([0.15, 0.15, 1., 1., 1.], dtype=np.float32),
        y_short,
        np.zeros_like(y_short),
    ))
    current_offset = np.column_stack((
        np.full_like(y_long, 3.), y_long, np.zeros_like(y_long)))
    history_offset = np.column_stack((
        np.full_like(y_long, 3.21), y_long, np.zeros_like(y_long)))

    def record(track_id, ground):
        return {
            'track_id': track_id,
            'ground': ground,
            'global': ground.copy(),
            'category': 1,
            'fit_status': 'first_fit',
        }

    current_records = [
        record(11, current_partial),
        record(22, current_offset),
    ]
    history_records = [
        record(11, history_partial),
        record(22, history_offset),
    ]
    current_label = dict(frame='current')
    history_label = dict(frame='history')

    dataset._build_calib = lambda label: {}
    dataset._global_from_ground = lambda label, meta: np.eye(4, dtype=np.float32)
    dataset._parse_lane_records = (
        lambda label, meta=None, is_history=False:
        current_records if label is current_label else history_records)
    dataset._clip_current_roi_and_fov = lambda points, calib: points
    empty_uv = np.zeros((0, 2), dtype=np.float32)
    empty_uvy = np.zeros((0, 3), dtype=np.float32)
    packed = dataset._pack_lane_targets(
        [current_partial, current_offset],
        [empty_uv, empty_uv],
        [empty_uvy, empty_uvy],
        [1, 1],
        track_ids=[11, 22],
    )
    packed.pop('gt_lanes_memory_mask')
    dataset._parse_openlane_gt = lambda label: packed

    enhanced = dataset._build_temporal_enhanced_gt(
        [history_label, current_label], [{}, {}])
    accepted_by_track = {
        association['track_id']: association['accepted']
        for association in enhanced['temporal_gt_associations']
        if 'reason' not in association
    }

    assert accepted_by_track == {11: True, 22: False}


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


def test_lidar_history_points_are_aligned_to_current_ego_before_concat(tmp_path):
    lidar_file = tmp_path / 'history.npy'
    np.save(lidar_file, np.array([[5., 2., 0., .5, .1, 0.]], dtype=np.float32))
    history_pose = np.eye(4, dtype=np.float32)
    history_pose[0, 3] = 10.
    current_pose = np.eye(4, dtype=np.float32)
    current_pose[0, 3] = 13.
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.num_lidar_feat = 6

    points = dataset._load_lidar_points(
        dict(lidar_file=lidar_file, pose=history_pose), current_pose,
        queue_index=0)

    # Raw Waymo [forward, left, up] -> current raw [2, 2, 0] ->
    # OpenLane ground [right, forward, up] = [-2, 2, 0].
    assert np.allclose(points[0, :3], [-2., 2., 0.])
    assert points[0, -1] == 0.


def test_temporal_pose_transform_is_expressed_in_ground_bev_axes():
    vehicle_transform = np.eye(4, dtype=np.float32)
    vehicle_transform[:3, 3] = [5., 2., 0.]

    ground_transform = _vehicle_transform_to_ground(vehicle_transform)

    assert np.allclose(ground_transform[:3, 3], [-2., 5., 0.])


def test_camera_lane_transform_uses_the_same_ground_axis_convention():
    cam_to_ground = _build_cam_to_ground_lane(np.eye(4, dtype=np.float32))

    assert np.allclose(cam_to_ground[:3, 0], [0., 1., 0.])
    assert np.allclose(cam_to_ground[:3, 1], [-1., 0., 0.])


def test_lidar_queue_does_not_duplicate_a_padded_physical_sweep(tmp_path):
    lidar_file = tmp_path / 'current.npy'
    np.save(lidar_file, np.array([[5., 0., 0., .5, .1, 0.]], dtype=np.float32))
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.num_lidar_feat = 6
    dataset.point_cloud_range = None
    dataset.shuffle_points = False
    pose = np.eye(4, dtype=np.float32)
    scene = dict(frames={
        'same': dict(lidar_file=lidar_file, pose=pose),
    })

    points = dataset._load_lidar_queue(scene, ['same', 'same', 'same'])

    assert points.shape == (1, 6)
    assert points[0, -1] == 2.


def test_openlane_getitem_loads_multimodal_temporal_sample():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(max_samples=1)
    data = dataset[0]
    len_queue = dataset.history_num_frames + 1
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
    if dataset.temporal_gt_augmentation:
        assert data['gt_lanes_category_enhanced'].shape == (cfg.max_lanes,)
    assert data['ground_lanes'].shape[0] == cfg.max_lanes
    assert dataset.metainfo['dataset_name'] == 'OpenLane'
    assert dataset.metainfo['classes'] == tuple(cfg.class_names)
    assert dataset.metainfo['num_category'] == cfg.num_category

    img_metas = data['img_metas']
    assert img_metas['num_history_frames'] == len_queue - 1
    assert img_metas['sampling_mode'] in {
        'distance', 'distance_fallback_offsets', 'neighbor'}
    assert len(img_metas['frame_keys']) == len_queue
    assert len(img_metas['queue']) == len_queue
    assert img_metas['num_views'] == cfg.num_views
    assert img_metas['image_augmentation_mode'] == (
        cfg.image_augmentation_mode)
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
            len_queue = dataset.history_num_frames + 1

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

    assert batch['img'].shape == (
        1, dataset.history_num_frames + 1, cfg.num_views, 3, img_h, img_w)
    assert batch['view_mask'].shape == (
        1, dataset.history_num_frames + 1, cfg.num_views)
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


def test_openlane_none_augmentation_mode_repeats_raw_views():
    np.random.seed(0)
    torch.manual_seed(0)

    dataset, cfg = _build_train_dataset(
        max_samples=1,
        dataset_cfg_overrides=dict(
            num_views=6,
            image_augmentation_mode='none',
        ))
    data = dataset[0]

    assert data['img_metas']['view_mask'].shape == (
        dataset.history_num_frames + 1, dataset.num_views)
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
        current_transform = queue[dataset.history_num_frames][
            'lidaradj2lidarcurr']
        assert np.allclose(current_transform, np.eye(4, dtype=np.float32), atol=1e-4)

        history_transforms = [
            queue[queue_idx]['lidaradj2lidarcurr']
            for queue_idx in range(dataset.history_num_frames)
        ]
        assert any(
            not np.allclose(transform, np.eye(4, dtype=np.float32), atol=1e-4)
            for transform in history_transforms
        )
        break
    else:
        pytest.skip('No non-duplicated history frames found in the sampled OpenLane subset.')
