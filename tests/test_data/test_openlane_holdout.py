from pathlib import Path

import numpy as np
import pytest
from mmengine.config import Config

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset


OPENLANE_TEMPORAL_CONFIG = (
    Path(__file__).resolve().parents[2]
    / 'configs/main/laneloom_openlane_image_lidar_temporal.py'
)


@pytest.mark.parametrize('count', [0, 1, 1000, 100000])
def test_point_shuffle_preserves_order_and_random_state(count):
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.point_cloud_range = None
    dataset.shuffle_points = True
    points = np.arange(count * 5, dtype=np.float32).reshape(count, 5)
    original_state = np.random.get_state()
    try:
        np.random.seed(7)
        expected = points.copy()
        np.random.shuffle(expected)
        expected_state = np.random.get_state()
        np.random.seed(7)
        actual = dataset._process_points(points.copy())
        actual_state = np.random.get_state()
        np.testing.assert_array_equal(actual, expected)
        assert actual_state[0] == expected_state[0]
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]
    finally:
        np.random.set_state(original_state)


def test_resume_rejects_point_alignment_change():
    from LaneLoom.tools.train_q0_prior import (
        model_training_signature, training_dataset_signature, validate_resume_protocol)
    cfg = Config.fromfile(OPENLANE_TEMPORAL_CONFIG)
    protocol = dict(training_labels_sha256='same labels', excluded_segments=[],
                    batch_size=8, model_type='LiDARShapeTokenPrior', epochs=2, seed=7,
                    training_dataset_sha256=training_dataset_signature(cfg),
                    q0_model_sha256=model_training_signature(cfg))
    saved = dict(meta=dict(protocol=protocol, cfg=cfg.pretty_text))
    cfg.model.pts_bev_encoder.align_to_lane_ground = True
    updated = dict(protocol, q0_model_sha256=model_training_signature(cfg))
    with pytest.raises(ValueError, match='model, loss settings'):
        validate_resume_protocol(saved, updated)


def test_segment_exclusion_precedes_sample_limits_and_covers_neighbors(tmp_path):
    for segment in ('segment-heldout', 'segment-training'):
        directory = tmp_path / 'training' / segment
        directory.mkdir(parents=True)
        for frame in ('000', '001', '002'):
            (directory / (frame + '.json')).write_text('{}')
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.label_root = tmp_path
    dataset.split = 'training'
    dataset.load_interval = 1
    dataset.excluded_segments = frozenset(['segment-heldout'])
    labels = dataset._load_label_files(max_samples=2)
    assert len(labels) == 2
    assert all(p.parent.name == 'segment-training' for p in labels)
    dataset.excluded_segments = frozenset()
    assert len(dataset._load_label_files()) == 6


def test_bounded_scene_cache_evicts_metadata_without_changing_values():
    dataset = OpenLaneDataset.__new__(OpenLaneDataset)
    dataset.scene_cache_size = 2
    cache = {}
    a, b, c = object(), object(), object()
    assert dataset._cache_scene(cache, 'a', a) is a
    dataset._cache_scene(cache, 'b', b)
    dataset._cache_scene(cache, 'c', c)
    assert set(cache) == {'b', 'c'}
    assert cache['b'] is b and cache['c'] is c


def test_resume_rejects_silent_supervision_change_and_accepts_original_config():
    from LaneLoom.tools.train_q0_prior import (
        training_dataset_signature, model_training_signature, validate_resume_protocol)
    cfg = Config.fromfile(OPENLANE_TEMPORAL_CONFIG)
    protocol = dict(training_labels_sha256='same labels',
                    excluded_segments=[], batch_size=8,
                    model_type='LiDARShapeTokenPrior', epochs=10, seed=7)
    # Older saved checkpoints predate the dataset fingerprint.
    saved = dict(meta=dict(protocol=protocol.copy(), cfg=cfg.pretty_text))
    protocol['training_dataset_sha256'] = training_dataset_signature(cfg)
    protocol['q0_model_sha256'] = model_training_signature(cfg)
    validate_resume_protocol(saved, protocol)
    cfg.train_dataloader.dataset.dataset_cfg.full_shape_visible_fit_tolerance_m = .01
    protocol['training_dataset_sha256'] = training_dataset_signature(cfg)
    with pytest.raises(ValueError, match='supervision differs'):
        validate_resume_protocol(saved, protocol)


def test_resume_rejects_silent_auxiliary_loss_change():
    from LaneLoom.tools.train_q0_prior import (
        training_dataset_signature, model_training_signature, validate_resume_protocol)
    cfg = Config.fromfile(OPENLANE_TEMPORAL_CONFIG)
    cfg.q0_raw_visible_supervision = True
    cfg.q0_lane_field_loss_weight = 0.
    old = dict(training_labels_sha256='same labels', excluded_segments=[],
               batch_size=8, model_type='LiDARShapeTokenPrior', epochs=10, seed=7,
               training_dataset_sha256=training_dataset_signature(cfg))
    saved = dict(meta=dict(protocol=old.copy(), cfg=cfg.pretty_text))
    cfg.q0_lane_field_loss_weight = .1
    old['q0_model_sha256'] = model_training_signature(cfg)
    with pytest.raises(ValueError, match='loss settings'):
        validate_resume_protocol(saved, old)
