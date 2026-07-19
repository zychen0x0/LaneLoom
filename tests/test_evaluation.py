import numpy as np
import pytest
import torch
from datetime import datetime
from types import SimpleNamespace

import LaneFormer  # noqa: F401 - register the project metric.
import LaneFormer.engine.hooks.laneformer_logger_hook as logger_module
from mmengine.config import Config
from mmengine.registry import METRICS
from LaneFormer.engine.experiment import configure_train_experiment
from LaneFormer.engine.openlane_test_cases import build_openlane_test_configs
from LaneFormer.engine.hooks import LaneFormerLoggerHook
from LaneFormer.engine.runner import LaneFormerRunner


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


def test_anchor_metric_uses_assigned_indices_for_category_accuracy():
    metric = METRICS.build(dict(type='OpenLaneAnchorMetric', prefix=None))
    sample = _sample()
    second_lane = sample['gt_lanes_3d'][0].clone()
    second_lane[:, 0] += 10.0
    sample['gt_lanes_3d'][1] = second_lane
    sample['gt_lanes_visibility'][1] = 1.0
    sample['gt_lanes_category'] = torch.tensor([2, 7])
    sample['pred_x'] = torch.stack((
        sample['gt_lanes_3d'][1, :, 0],
        sample['gt_lanes_3d'][0, :, 0]))
    sample['pred_z'] = torch.stack((
        sample['gt_lanes_3d'][1, :, 2],
        sample['gt_lanes_3d'][0, :, 2]))
    sample['pred_visibility'] = torch.ones(2, 4)
    sample['pred_scores'] = torch.ones(2)
    sample['pred_category'] = torch.tensor([7, 2])

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Category Accuracy'] == 1.0


def test_anchor_metric_excludes_queries_suppressed_by_curve_nms():
    metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, score_threshold=0.05))
    sample = _sample()
    sample['pred_x'] = sample['pred_x'].repeat(2, 1)
    sample['pred_z'] = sample['pred_z'].repeat(2, 1)
    sample['pred_visibility'] = sample['pred_visibility'].repeat(2, 1)
    sample['pred_scores'] = torch.tensor([0.2, 0.1])
    sample['pred_category'] = torch.tensor([2, 2])
    sample['pred_nms_keep_queries'] = torch.tensor([True, False])

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['num_pred'] == 1.0
    assert values['num_pred_before_nms'] == 2.0
    assert values['num_nms_suppressed'] == 1.0
    assert values['nms_suppression_ratio'] == .5
    assert values['num_tp'] == 1.0
    assert values['F1'] == 1.0


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


def test_anchor_metric_uses_temporal_supplement_category():
    sample = _sample(category=2, predicted_category=7)
    sample['gt_lanes_3d_enhanced'] = sample['gt_lanes_3d'].clone()
    sample['gt_lanes_visibility_enhanced'] = sample['gt_lanes_visibility'].clone()
    sample['gt_lanes_category_enhanced'] = torch.tensor([7, 0])
    metric = METRICS.build(dict(type='OpenLaneAnchorMetric', prefix=None))

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Category Accuracy'] == 1.0


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


@pytest.mark.parametrize(
    ('selection', 'expected'),
    [
        ('validation', ['validation']),
        ('test', [
            'up_down_case', 'curve_case', 'extreme_weather_case',
            'intersection_case', 'merge_split_case', 'night_case']),
        ('curve_case', ['curve_case']),
    ])
def test_openlane_test_configs_filter_requested_cases(selection, expected):
    cfg = Config.fromfile(
        '/workspace/LaneFormer/configs/laneformer_openlane_image_lidar.py')

    case_configs = build_openlane_test_configs(cfg, selection)

    assert [name for name, _ in case_configs] == expected


def test_openlane_test_configs_reject_unknown_case():
    cfg = Config.fromfile(
        '/workspace/LaneFormer/configs/laneformer_openlane_image_lidar.py')

    with pytest.raises(ValueError, match='Unknown OpenLane test case'):
        build_openlane_test_configs(cfg, 'unknown')


def test_logger_reports_validation_progress_and_final_lane_metrics(monkeypatch):
    messages = []
    runner = SimpleNamespace(
        rank=0,
        iter=123,
        val_dataloader=[None] * 100,
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(add_scalars=lambda *_args, **_kwargs: None),
    )
    hook = LaneFormerLoggerHook(interval=50)
    clock = iter([0.0, 98.0])
    monkeypatch.setattr(logger_module.time, 'monotonic', lambda: next(clock))

    hook.before_val_epoch(runner)
    hook.after_val_iter(runner, batch_idx=49)
    hook.after_val_epoch(runner, metrics=dict(
        **{'OpenLane/F1': 0.5, 'OpenLane/Recall': 0.4,
           'OpenLane/Precision': 0.6, 'OpenLane/Category Accuracy': 0.7},
        **{'OpenLane/x_error_near': 0.1, 'OpenLane/z_error_near': 0.2,
           'OpenLane/x_error_far': 0.3, 'OpenLane/z_error_far': 0.4},
        **{'OpenLane/num_gt': 100.0, 'OpenLane/num_pred': 120.0,
           'OpenLane/num_tp': 40.0},
    ))

    progress = next(message for message in messages if message.startswith('Val:'))
    assert 'Val: [50/100] 50.00%' in progress
    assert 'Elapsed/Expected 00:01:38 / 00:03:16    ETA 00:01:38' in progress
    assert 'Time 1.96 s/iter' in progress
    summary = next(message for message in messages if 'Evaluation laneline F1' in message)
    assert 'F1: 0.50000000' in summary
    assert 'Recall: 0.40000000' in summary
    assert 'Precision: 0.60000000' in summary
    assert 'Category Accuracy: 0.70000000' in summary
    assert 'GT / Pred / TP: 100 / 120 / 40' in summary


def test_logger_reports_test_case_progress_eta_and_final_batch(monkeypatch):
    messages = []
    runner = SimpleNamespace(
        rank=0,
        work_dir='/tmp/nms_evaluation/curve_case/logs',
        test_dataloader=[None] * 320,
        logger=SimpleNamespace(info=messages.append),
    )
    hook = LaneFormerLoggerHook(interval=50)
    clock = iter([0.0, 88.0, 640.0])
    monkeypatch.setattr(logger_module.time, 'monotonic', lambda: next(clock))

    hook.before_test_epoch(runner)
    hook.after_test_iter(runner, batch_idx=49)
    hook.after_test_iter(runner, batch_idx=319)

    assert messages[0].startswith('Test curve_case: [50/320] 15.62%')
    assert 'Elapsed/Expected 00:01:28 / 00:09:23    ETA 00:07:55' in messages[0]
    assert 'Time 1.76 s/iter' in messages[0]
    assert messages[1].startswith('Test curve_case: [320/320] 100.00%')
    assert 'ETA 00:00:00' in messages[1]


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


def test_laneformer_runner_writes_log_directly_in_work_dir(tmp_path):
    logs_dir = tmp_path / 'logs'
    runner = LaneFormerRunner(
        model=torch.nn.Linear(1, 1),
        work_dir=str(logs_dir),
        default_scope=None,
        experiment_name='direct_log_success',
    )

    assert runner.log_dir == str(logs_dir)
    assert len(list(logs_dir.glob('*.log'))) == 1
    assert not [path for path in logs_dir.iterdir() if path.is_dir()]


def test_laneformer_runner_does_not_create_log_subdir_when_init_fails(tmp_path):
    logs_dir = tmp_path / 'logs'

    with pytest.raises(KeyError):
        LaneFormerRunner(
            model=dict(type='MissingLaneFormerModel'),
            work_dir=str(logs_dir),
            default_scope=None,
            experiment_name='direct_log_failure',
        )

    assert not [path for path in logs_dir.iterdir() if path.is_dir()]
