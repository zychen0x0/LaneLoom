import json

import numpy as np
import pytest
import torch
from mmengine.config import Config

from LaneLoom.datasets.once.dataset import CAMERA_TO_GROUND, ONCEDataset
from LaneLoom.models.backbones.dynamic_pillar_bev import RGBPointPainter


def test_empty_labels_are_excluded_from_current_and_history_frames(tmp_path):
    cfg = Config.fromfile('configs/_base_/datasets/once_datasets.py')
    dataset_cfg = dict(cfg.dataset_cfg)
    dataset_cfg.update(
        data_label_dir=str(tmp_path), split_dir=str(tmp_path / 'list'),
        info_dir=str(tmp_path / 'infos'),
        lidar_processed_dir=str(tmp_path / 'lidars'))
    sequence = '000001'
    label_dir = tmp_path / 'train' / sequence / 'cam01'
    label_dir.mkdir(parents=True)
    split_dir = tmp_path / 'list'
    split_dir.mkdir()
    keys = [str(1616000000000 + index * 1000) for index in range(4)]
    (split_dir / 'train.txt').write_text(''.join(
        f'/{sequence}/cam01/{key}.jpg\n' for key in keys))
    for index, key in enumerate(keys):
        (label_dir / f'{key}.json').write_text(
            '' if index == 1 else json.dumps({'lanes': []}))
    lidar_dir = tmp_path / 'lidars' / sequence / 'lidar_roof'
    lidar_dir.mkdir(parents=True)
    info_dir = tmp_path / 'infos' / sequence
    info_dir.mkdir(parents=True)
    frames = []
    for index, key in enumerate(keys):
        np.array([[0, 2, 0, 1]], dtype=np.float32).tofile(
            lidar_dir / f'{key}.bin')
        frames.append(dict(frame_id=key, pose=[0, 0, 0, 1, index * 5, 0, 0]))
    (info_dir / f'{sequence}.json').write_text(json.dumps(dict(
        calib=dict(cam01=dict(cam_to_velo=np.eye(4).tolist())), frames=frames)))
    with pytest.warns(UserWarning, match='skipped 1 empty labels'):
        dataset = ONCEDataset(
            dataset_cfg=dataset_cfg, temporal_sampling_distance=(10., 5.),
            smooth_3d_gt=False, gt_outlier_residual=.75,
            modality=cfg.input_modality, split='train')
    assert len(dataset) == 3
    assert not dataset.temporal_gt_augmentation
    label_scene = dataset._load_label_scene(sequence)
    assert keys[1] not in label_scene['frames']
    queue, mode, _, _ = dataset._sample_frame_queue_keys(
        sequence, keys[-1], label_scene)
    assert queue == [keys[0], keys[2], keys[3]]
    assert mode == 'distance'
    queue, _, _, _ = dataset._sample_frame_queue_keys(
        sequence, keys[0], label_scene)
    assert queue == [keys[0]] * 3
    # Padding must not triple the first physical sweep.
    assert dataset._load_lidar_queue(
        dataset._load_lidar_scene(sequence), queue).shape[0] == 1


def test_once_history_alignment_matches_point_alignment():
    dataset = ONCEDataset.__new__(ONCEDataset)
    dataset.org_h, dataset.org_w = 1020, 1920
    dataset.resize_h, dataset.resize_w = 510, 960
    dataset.num_views = 1
    dataset.mean = np.zeros((3, 1, 1), dtype=np.float32)
    dataset.std = np.ones((3, 1, 1), dtype=np.float32)
    history_pose = np.eye(4, dtype=np.float32)
    history_pose[:3, 3] = [3, 4, 5]
    current_pose = np.eye(4, dtype=np.float32)
    current_pose[:3, 3] = [7, 8, 9]
    lidar_to_ground = CAMERA_TO_GROUND.copy()
    lidar_to_ground[:3, 3] = [1, 2, 3]
    key = '1616000000000'
    scene = dict(frames={key: dict(
        pose=history_pose, lidar_to_ground=lidar_to_ground,
        lidar_file='points.bin', timestamp_micros=int(key))})
    label = dict(calibration=np.eye(3, 4).tolist())
    meta = dataset._build_frame_meta(
        '000001', key, label, 'image.jpg', scene, current_pose)
    raw_point = np.array([2, 5, 7, 1], dtype=np.float32)
    expected = (lidar_to_ground @ np.linalg.inv(current_pose)
                @ history_pose @ raw_point)
    actual = meta['lidaradj2lidarcurr'] @ lidar_to_ground @ raw_point
    np.testing.assert_allclose(actual, expected)
    assert meta['timestamp'] == int(key) / 1000


def test_once_pointpainting_uses_each_history_image_in_ground_axes():
    painter = RGBPointPainter(point_features=[
        'x', 'y', 'z', 'intensity', 'queue_index'])
    images = torch.zeros(1, 2, 1, 3, 2, 2)
    images[:, 0] = .25
    images[:, 1] = .75
    history_to_current = torch.eye(4)
    history_to_current[0, 3] = 5
    projection = torch.tensor([
        [1., 0, 0, 0], [0, 0, -1., 0],
        [0, 1., 0, 0], [0, 0, 0, 1.]])
    queue = {index: dict(
        lane2img=[projection], lidar2img=[projection],
        lidaradj2lidarcurr=transform)
        for index, transform in enumerate((history_to_current, torch.eye(4)))}
    points = torch.tensor([[5., 2, 0, 1, 0], [0, 2., 0, 1, 1]])
    painted = painter.paint_points(
        [points], images, [dict(queue=queue)])[0]
    torch.testing.assert_close(painted[:, 4:7], torch.tensor([
        [.25, .25, .25], [.75, .75, .75]]))
    torch.testing.assert_close(painted[:, -1], points[:, -1])
