from pathlib import Path

import numpy as np
import pytest
from mmengine.config import Config

from LaneLoom.datasets.openlane.dataset import OpenLaneDataset


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


def test_aligned_comparison_uses_same_calibrated_input_and_rejects_legacy_resume():
    from LaneLoom.tools.train_q0_prior import (
        model_training_signature, training_dataset_signature, validate_resume_protocol)
    old = Config.fromfile('configs/ablations/laneloom_openlane_q0_raw_visible.py')
    baseline = Config.fromfile('configs/ablations/laneloom_openlane_q0_aligned.py')
    candidate = Config.fromfile('configs/ablations/laneloom_openlane_q0_aligned_field_curve.py')
    assert baseline.model.pts_bev_encoder.align_to_lane_ground
    assert baseline.model.pts_bev_encoder == candidate.model.pts_bev_encoder
    assert baseline.train_dataloader == candidate.train_dataloader == old.train_dataloader
    assert baseline.q0_target_frames == candidate.q0_target_frames == old.q0_target_frames
    assert len(baseline.q0_excluded_segments) == 19
    assert model_training_signature(baseline) != model_training_signature(old)
    protocol = dict(training_labels_sha256='same labels', excluded_segments=old.q0_excluded_segments,
                    batch_size=8, model_type='LiDARShapeTokenPrior', epochs=2, seed=7,
                    training_dataset_sha256=training_dataset_signature(old),
                    q0_model_sha256=model_training_signature(old))
    saved = dict(meta=dict(protocol=protocol, cfg=old.pretty_text))
    updated = dict(protocol, q0_model_sha256=model_training_signature(baseline))
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


def test_spatial_comparison_changes_only_template_extractor():
    root = Path(__file__).resolve().parents[2] / 'configs/ablations'
    baseline = Config.fromfile(root / 'laneloom_openlane_q0_holdout.py')
    spatial = Config.fromfile(root / 'laneloom_openlane_q0_spatial.py')
    assert spatial.train_dataloader == baseline.train_dataloader
    assert len(spatial.q0_excluded_segments) == 19
    assert sum(map(len, spatial.q0_target_frames.values())) == 21
    assert spatial.load_from is None
    assert spatial.model.lane_head.anchor_prior.max_templates == 5
    assert spatial.model.lane_head.proposal_expander.num_x_seeds == 101
    assert spatial.model.lane_head.anchor_prior.type == 'SpatialBEVShapePrior'
    old = baseline.model.to_dict()
    new = spatial.model.to_dict()
    del old['lane_head']['anchor_prior']
    del new['lane_head']['anchor_prior']
    assert old == new


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
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/ablations/laneloom_openlane_q0_holdout.py')
    protocol = dict(training_labels_sha256='same labels',
                    excluded_segments=cfg.q0_excluded_segments, batch_size=8,
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


def test_continuous_readout_keeps_target_holdout_and_505_query_budget():
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/ablations/laneloom_openlane_q0_continuous.py')
    assert len(cfg.q0_excluded_segments) == 19
    assert sum(map(len, cfg.q0_target_frames.values())) == 21
    assert cfg.train_dataloader.dataset.dataset_cfg.excluded_segments == cfg.q0_excluded_segments
    assert cfg.train_dataloader.dataset.dataset_cfg.full_shape_visible_fit_tolerance_m == .01
    assert cfg.model.pts_bev_encoder.return_road_evidence
    assert cfg.model.lane_head.anchor_prior.max_templates == 5
    assert cfg.model.lane_head.proposal_expander.num_x_seeds == 101
    assert cfg.model.lane_head.anchor_prior.direction_loss_weight == 0
    assert cfg.model.lane_head.anchor_prior.curvature_loss_weight == 0
    assert cfg.load_from is None


def test_resume_rejects_silent_auxiliary_loss_change():
    from LaneLoom.tools.train_q0_prior import (
        training_dataset_signature, model_training_signature, validate_resume_protocol)
    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(root / 'configs/ablations/laneloom_openlane_q0_continuous.py')
    old = dict(training_labels_sha256='same labels', excluded_segments=cfg.q0_excluded_segments,
               batch_size=8, model_type='ContinuousBEVShapePrior', epochs=10, seed=7,
               training_dataset_sha256=training_dataset_signature(cfg))
    saved = dict(meta=dict(protocol=old.copy(), cfg=cfg.pretty_text))
    cfg.model.lane_head.anchor_prior.direction_loss_weight = .1
    old['q0_model_sha256'] = model_training_signature(cfg)
    with pytest.raises(ValueError, match='loss settings'):
        validate_resume_protocol(saved, old)


def test_raw_visible_comparison_keeps_holdout_and_changes_only_extractor():
    from LaneLoom.tools.train_q0_prior import model_training_signature
    root = Path(__file__).resolve().parents[2] / 'configs/ablations'
    baseline = Config.fromfile(root / 'laneloom_openlane_q0_raw_visible.py')
    spatial = Config.fromfile(root / 'laneloom_openlane_q0_raw_visible_spatial.py')
    assert baseline.q0_raw_visible_supervision and spatial.q0_raw_visible_supervision
    assert not baseline.train_dataloader.dataset.dataset_cfg.full_shape_template_targets
    assert not baseline.model.lane_head.anchor_prior.supervise_shapes
    assert len(baseline.q0_excluded_segments) == 19
    assert baseline.train_dataloader == spatial.train_dataloader
    old, new = baseline.model.to_dict(), spatial.model.to_dict()
    del old['lane_head']['anchor_prior'], new['lane_head']['anchor_prior']
    assert old == new
    signature = model_training_signature(baseline)
    baseline.q0_raw_visible_supervision = False
    assert model_training_signature(baseline) != signature
