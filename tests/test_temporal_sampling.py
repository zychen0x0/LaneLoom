from pathlib import Path
import shutil

import numpy as np
import pytest
from mmengine.config import Config

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset
from LaneLoom.utils.temporal_sampling import resolve_temporal_sampling


@pytest.mark.parametrize(
    ('value', 'targets', 'fallbacks', 'frames', 'queue_length', 'use_temporal_gt'),
    [
        (False, (), (), (0,), 1, False),
        ((10.0, 5.0), (10.0, 5.0), (10, 5), (-10, -5, 0), 3, True),
        ((2, 1), (2.0, 1.0), (2, 1), (-2, -1, 0), 3, True),
        (
            (15, 10, 5),
            (15.0, 10.0, 5.0),
            (15, 10, 5),
            (-15, -10, -5, 0),
            4,
            True,
        ),
    ],
)
def test_temporal_sampling_is_derived_from_one_public_value(
        value, targets, fallbacks, frames, queue_length, use_temporal_gt):
    sampling = resolve_temporal_sampling(value)

    assert sampling.distance_targets_m == targets
    assert sampling.fallback_frame_offsets == fallbacks
    assert sampling.frame_offsets == frames
    assert sampling.history_num_frames == len(targets)
    assert sampling.queue_length == queue_length
    assert sampling.temporal_gt_augmentation is use_temporal_gt


@pytest.mark.parametrize(
    'value',
    [True, None, (), [10, 5], (5, 10), (5, 5), (0,), (-1,), (2.5, 1)],
)
def test_temporal_sampling_rejects_ambiguous_values(value):
    with pytest.raises((TypeError, ValueError)):
        resolve_temporal_sampling(value)


def test_openlane_config_uses_one_temporal_sampling_value():
    cfg = Config.fromfile(
        '/workspace/LaneLoom/configs/main/laneloom_openlane_image_lidar.py')

    assert cfg.temporal_sampling_distance == (10.0, 5.0)
    assert cfg.train_dataloader.dataset.temporal_sampling_distance == (
        10.0, 5.0)
    assert cfg.val_dataloader.dataset.temporal_sampling_distance == (10.0, 5.0)
    assert cfg.test_dataloader.dataset.temporal_sampling_distance == (10.0, 5.0)
    assert cfg.model.pts_bev_encoder.temporal_sampling_distance == (10.0, 5.0)
    assert cfg.val_evaluator.use_temporal_gt_supplement is True

    legacy_fields = {
        'frames',
        'history_num_frames',
        'history_distance_m',
        'history_distance_step_m',
        'history_fallback_frame_offsets',
        'temporal_gt_augmentation',
        'queue_length',
    }
    assert legacy_fields.isdisjoint(cfg.keys())


def test_false_switches_the_entire_openlane_config_to_single_frame(tmp_path):
    config_root = Path(__file__).resolve().parents[1] / 'configs'
    copied_root = tmp_path / 'configs'
    shutil.copytree(config_root, copied_root)
    dataset_config = copied_root / '_base_/datasets/openlane_datasets.py'
    source = dataset_config.read_text()
    dataset_config.write_text(source.replace(
        'temporal_sampling_distance = (10.0, 5.0)',
        'temporal_sampling_distance = False'))

    cfg = Config.fromfile(
        copied_root / 'main/laneloom_openlane_image_lidar.py')

    assert cfg.temporal_sampling_distance is False
    assert cfg.train_dataloader.dataset.temporal_sampling_distance is False
    assert cfg.val_dataloader.dataset.temporal_sampling_distance is False
    assert cfg.test_dataloader.dataset.temporal_sampling_distance is False
    assert cfg.model.pts_bev_encoder.temporal_sampling_distance is False
    assert cfg.val_evaluator.prefix == 'OpenLaneOfficial'
    assert cfg.val_evaluator.use_temporal_gt_supplement is False
    assert cfg.default_hooks.checkpoint.save_best == 'OpenLaneOfficial/F1'


def test_openlane_samples_each_explicit_distance_target():
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    sampling = resolve_temporal_sampling((2, 1))
    dataset.history_num_frames = sampling.history_num_frames
    dataset.history_distance_targets_m = sampling.distance_targets_m

    def pose(x):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = x
        return matrix

    scene = dict(
        keys=['0', '1', '2', '3'],
        frames={
            '0': dict(pose=pose(0)),
            '1': dict(pose=pose(1)),
            '2': dict(pose=pose(2)),
            '3': dict(pose=pose(3)),
        },
    )

    assert dataset._sample_history_by_distance(scene, '3') == ['1', '2']


def test_openlane_falls_back_only_the_unsatisfied_distance_slot():
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    sampling = resolve_temporal_sampling((10, 5))
    dataset.history_num_frames = sampling.history_num_frames
    dataset.history_frame_step = 1
    dataset.history_distance_targets_m = sampling.distance_targets_m
    dataset.history_fallback_frame_offsets = (
        sampling.fallback_frame_offsets)

    keys = [f'{index:02d}' for index in range(13)]

    def pose(x):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = x
        return matrix

    lidar_scene = dict(
        keys=keys,
        frames={key: dict(pose=pose(8)) for key in keys},
    )
    lidar_scene['frames']['09']['pose'] = pose(7)
    lidar_scene['frames']['12']['pose'] = pose(12)
    label_scene = dict(keys=keys, frames={key: key for key in keys})
    dataset._load_lidar_scene = lambda _: lidar_scene

    queue, mode, _, _ = dataset._sample_frame_queue_keys(
        'scene', '12', label_scene=label_scene)

    assert queue == ['02', '09', '12']
    assert mode == 'distance_fallback_offsets'


def test_frame_fallback_uses_the_unfiltered_label_timeline():
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.history_fallback_frame_offsets = (2,)
    label_scene = dict(
        keys=['0', '1', '2', '3'],
        frames={key: key for key in ('0', '1', '2', '3')},
    )

    assert dataset._sample_history_by_frame_offsets(
        label_scene, '3', available_keys={'0', '2', '3'}) == [None]
    assert dataset._sample_history_by_frame_offsets(
        label_scene, '3', available_keys=label_scene['keys']) == ['1']


def test_missing_lidar_uses_inferred_frame_fallbacks():
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    sampling = resolve_temporal_sampling((10, 5))
    dataset.history_num_frames = sampling.history_num_frames
    dataset.history_frame_step = 1
    dataset.history_distance_targets_m = sampling.distance_targets_m
    dataset.history_fallback_frame_offsets = (
        sampling.fallback_frame_offsets)
    dataset._load_lidar_scene = lambda _: None
    keys = [f'{index:02d}' for index in range(13)]
    label_scene = dict(keys=keys, frames={key: key for key in keys})

    queue, mode, lidar_scene, _ = dataset._sample_frame_queue_keys(
        'scene', '12', label_scene=label_scene)

    assert queue == ['02', '07', '12']
    assert mode == 'distance_fallback_offsets'
    assert lidar_scene is None
