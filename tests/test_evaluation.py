import numpy as np
import torch
from datetime import datetime
from types import SimpleNamespace

import LaneFormer  # noqa: F401 - register the project metric.
from mmengine.config import Config
from mmengine.registry import METRICS
from LaneFormer.engine.experiment import configure_train_experiment
from LaneFormer.engine.openlane_test_cases import build_openlane_test_configs
from LaneFormer.engine.hooks import LaneFormerLoggerHook


def _sample(x_offset=0.0, z_offset=0.0, score=0.9, scene='rainy-turn',
            scenario='', category=2, predicted_category=2):
    anchors = torch.tensor([0.0, 20.0, 40.0, 60.0])
    gt = torch.zeros(2, 4, 3)
    gt[0, :, 0] = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    gt[0, :, 1] = anchors
    gt[0, :, 2] = 0.2
    gt_visibility = torch.zeros(2, 4)
    gt_visibility[0] = 1.0
    return dict(
        pred_x=(gt[0, :, 0] + x_offset).unsqueeze(0),
        pred_z=(gt[0, :, 2] + z_offset).unsqueeze(0),
        pred_visibility=torch.ones(1, 4),
        pred_scores=torch.tensor([score]),
        pred_category=torch.tensor([predicted_category]),
        anchor_y_steps=anchors,
        gt_lanes_3d=gt,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=torch.tensor([category, 0]),
        scene_name=scene,
        scenario_name=scenario,
    )


def test_anchor_metric_matches_a_good_lane_and_reports_axis_errors():
    metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, score_threshold=0.3,
        distance_threshold=0.5, match_ratio_threshold=0.75,
        near_distance=40.0))
    metric.process(None, [_sample(x_offset=0.1, z_offset=0.2)])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Recall'] == 1.0
    assert values['Precision'] == 1.0
    assert values['Category Accuracy'] == 1.0
    assert np.isclose(values['x_error_near'], 0.1)
    assert np.isclose(values['z_error_far'], 0.2)
    assert all('/F1' not in name for name in values if name != 'F1')


def test_anchor_metric_rejects_a_lane_with_insufficient_visible_overlap():
    metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, score_threshold=0.3,
        distance_threshold=0.5, match_ratio_threshold=0.75))
    sample = _sample()
    sample['pred_visibility'][0, :2] = 0.0
    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['num_gt'] == 1.0
    assert values['num_pred'] == 1.0
    assert values['num_tp'] == 0.0
    assert values['F1'] == 0.0
    assert values['Category Accuracy'] == 0.0


def test_anchor_metric_reports_category_accuracy_for_geometric_matches():
    metric = METRICS.build(dict(type='OpenLaneAnchorMetric', prefix=None))
    metric.process(None, [_sample(predicted_category=7)])

    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Category Accuracy'] == 0.0


def test_anchor_metric_uses_temporal_supplement_by_default_and_can_disable_it():
    sample = _sample()
    enhanced = sample['gt_lanes_3d'].clone()
    enhanced[0, :, 0] += 8.0
    sample['gt_lanes_3d_enhanced'] = enhanced
    sample['gt_lanes_visibility_enhanced'] = sample['gt_lanes_visibility'].clone()

    temporal_metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, distance_threshold=0.5))
    temporal_metric.process(None, [sample])
    assert temporal_metric.compute_metrics(temporal_metric.results)['F1'] == 0.0

    current_metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, distance_threshold=0.5,
        use_temporal_gt_supplement=False))
    current_metric.process(None, [sample])
    assert current_metric.compute_metrics(current_metric.results)['F1'] == 1.0


def test_config_enables_mmengine_validation_and_test_evaluators():
    cfg = Config.fromfile(
        '/workspace/LaneFormer/configs/laneformer_openlane_image_lidar.py')
    assert cfg.val_cfg.type == 'ValLoop'
    assert cfg.test_cfg.type == 'TestLoop'
    assert cfg.val_dataloader is not None
    assert cfg.test_dataloader is not None
    assert cfg.val_evaluator.type == 'OpenLaneAnchorMetric'
    assert cfg.test_evaluator.type == 'OpenLaneAnchorMetric'
    assert cfg.val_evaluator.use_temporal_gt_supplement is True
    assert cfg.test_dataloader.dataset.split == 'test'
    assert cfg.test_dataloader.dataset.dataset_cfg.load_interval == 1
    assert cfg.default_hooks.checkpoint.save_best == 'OpenLane/F1'
    assert cfg.default_hooks.checkpoint.rule == 'greater'


def test_openlane_test_configs_select_validation_and_each_official_case():
    cfg = Config.fromfile(
        '/workspace/LaneFormer/configs/laneformer_openlane_image_lidar.py')

    case_configs = build_openlane_test_configs(cfg)

    assert [name for name, _ in case_configs] == [
        'validation',
        'up_down_case',
        'curve_case',
        'extreme_weather_case',
        'intersection_case',
        'merge_split_case',
        'night_case',
    ]
    for name, case_cfg in case_configs:
        expected_split = 'validation' if name == 'validation' else f'test/{name}'
        dataset_cfg = case_cfg.test_dataloader.dataset.dataset_cfg
        assert case_cfg.test_dataloader.dataset.split == 'val'
        assert dataset_cfg.val_split == expected_split
        assert case_cfg.test_evaluator.prefix == f'OpenLane/{name}'

    assert cfg.test_dataloader.dataset.split == 'test'
    assert cfg.dataset_cfg.val_split == 'validation'


def test_logger_reports_validation_progress_and_final_lane_metrics():
    messages = []
    runner = SimpleNamespace(
        rank=0,
        iter=123,
        val_dataloader=[None] * 100,
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(add_scalars=lambda *_args, **_kwargs: None),
    )
    hook = LaneFormerLoggerHook(interval=50)

    hook.after_val_iter(runner, batch_idx=49)
    hook.after_val_epoch(runner, metrics=dict(
        **{'OpenLane/F1': 0.5, 'OpenLane/Recall': 0.4,
           'OpenLane/Precision': 0.6, 'OpenLane/Category Accuracy': 0.7},
        **{'OpenLane/x_error_near': 0.1, 'OpenLane/z_error_near': 0.2,
           'OpenLane/x_error_far': 0.3, 'OpenLane/z_error_far': 0.4},
    ))

    assert any('Val: [50/100]' in message for message in messages)
    summary = next(message for message in messages if 'Evaluation laneline F1' in message)
    assert 'F1: 0.50000000' in summary
    assert 'Recall: 0.40000000' in summary
    assert 'Precision: 0.60000000' in summary
    assert 'Category Accuracy: 0.70000000' in summary


def test_train_experiment_layout_uses_a_dated_laneformer_directory(tmp_path):
    cfg = Config(dict(
        work_dir=str(tmp_path / 'work_dirs' / 'laneformer'),
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1))))
    experiment_dir = configure_train_experiment(
        cfg, now=datetime(2026, 7, 12, 18, 30, 45))

    assert experiment_dir.name == 'exp_date_20260712_183045'
    assert (experiment_dir / 'logs').is_dir()
    assert (experiment_dir / 'Tensorboard').is_dir()
    assert (experiment_dir / 'checkpoints').is_dir()
    assert cfg.work_dir == str(experiment_dir / 'logs')
    assert cfg.default_hooks.checkpoint.type == 'LaneFormerCheckpointHook'
    assert cfg.default_hooks.checkpoint.out_dir == str(experiment_dir / 'checkpoints')
    assert [backend.type for backend in cfg.visualizer.vis_backends] == [
        'TensorboardVisBackend']
