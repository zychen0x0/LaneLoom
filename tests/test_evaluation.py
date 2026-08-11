import numpy as np
import pytest
import torch
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import LaneLoom  # noqa: F401 - register the project metric.
import LaneLoom.engine.hooks.laneloom_logger_hook as logger_module
from mmengine.config import Config
from mmengine.registry import METRICS
from LaneLoom.detectors import LaneLoom as LaneLoomModel
from LaneLoom.engine.experiment import configure_train_experiment
from LaneLoom.engine.openlane_test_cases import build_openlane_test_configs
from LaneLoom.engine.hooks import TrainingLoggerHook
from LaneLoom.engine.runner import ExperimentRunner
from LaneLoom.evaluation.lane_evaluation import OpenLaneLaneEvaluator


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


def _official_ground_lane(x_offset=0.0):
    y = np.array([3.0, 20.0, 40.0, 60.0, 80.0, 102.0])
    return np.column_stack((
        np.full_like(y, x_offset),
        y,
        np.zeros_like(y),
    ))


def _official_raw_label(ground_lane, category=2):
    vehicle_to_ground = np.array([
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., 0., 1.],
    ])
    ground_to_camera = np.array([
        [1., 0., 0.],
        [0., 0., 1.],
        [0., -1., 0.],
    ])
    rotation = (
        vehicle_to_ground
        @ np.linalg.inv(ground_to_camera)
        @ np.linalg.inv(vehicle_to_ground))
    camera_representation = np.array([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.],
    ])
    ground_h = np.column_stack(
        (ground_lane, np.ones(len(ground_lane))))
    camera_lane = (camera_representation @ ground_h.T)[:3]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    return dict(
        file_path='validation/segment/frame.jpg',
        extrinsic=extrinsic.tolist(),
        intrinsic=np.eye(3).tolist(),
        lane_lines=[dict(
            xyz=camera_lane.tolist(),
            visibility=np.ones(len(ground_lane)).tolist(),
            category=category)],
    )


def test_official_evaluator_uses_openlane_1p5m_matching_threshold():
    evaluator = OpenLaneLaneEvaluator()
    gt_lane = _official_ground_lane()
    prediction = _official_ground_lane(x_offset=1.0)

    summary = evaluator.evaluate_ground_lanes(
        [prediction], [2], [gt_lane], [np.ones(len(gt_lane))], [2])

    assert summary['recall_matches'] == 1
    assert summary['precision_matches'] == 1
    assert summary['match_count'] == 1


def test_official_metric_uses_current_raw_label_and_ignores_enhanced_gt():
    evaluator = OpenLaneLaneEvaluator()
    ground_lane = _official_ground_lane()
    raw_label = _official_raw_label(ground_lane)
    sample = _sample()
    sample.update(
        pred_x=torch.from_numpy(ground_lane[:, 0]).unsqueeze(0),
        pred_z=torch.from_numpy(ground_lane[:, 2]).unsqueeze(0),
        pred_visibility=torch.ones(1, len(ground_lane)),
        pred_scores=torch.tensor([0.9]),
        pred_category=torch.tensor([2]),
        anchor_y_steps=torch.from_numpy(ground_lane[:, 1]),
        raw_label=raw_label,
        gt_lanes_3d_enhanced=torch.full((1, len(ground_lane), 3), 999.0),
        gt_lanes_visibility_enhanced=torch.ones(1, len(ground_lane)),
    )
    metric = METRICS.build(dict(
        type='OpenLaneOfficialMetric', prefix=None, score_threshold=0.5,
        use_temporal_gt_supplement=False))

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Recall'] == 1.0
    assert values['Precision'] == 1.0


def test_official_metric_can_evaluate_temporal_supplement():
    ground_lane = _official_ground_lane()
    sample = _sample()
    sample.update(
        pred_x=torch.from_numpy(ground_lane[:, 0]).unsqueeze(0),
        pred_z=torch.from_numpy(ground_lane[:, 2]).unsqueeze(0),
        pred_visibility=torch.ones(1, len(ground_lane)),
        pred_scores=torch.tensor([0.9]),
        pred_category=torch.tensor([2]),
        anchor_y_steps=torch.from_numpy(ground_lane[:, 1]),
        raw_label=_official_raw_label(ground_lane),
        gt_lanes_3d_enhanced=torch.from_numpy(
            _official_ground_lane(x_offset=8.0)).unsqueeze(0),
        gt_lanes_visibility_enhanced=torch.ones(1, len(ground_lane)),
        gt_lanes_category_enhanced=torch.tensor([2]),
    )
    metric = METRICS.build(dict(
        type='OpenLaneOfficialMetric', prefix=None, score_threshold=0.5,
        use_temporal_gt_supplement=True))

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 0.0
    assert values['num_gt'] == 1.0


def test_evaluation_samples_expose_current_raw_label_to_official_metric():
    ground_lane = _official_ground_lane()
    raw_label = _official_raw_label(ground_lane)
    predictions = dict(
        x=torch.zeros(1, 1, 2),
        z=torch.zeros(1, 1, 2),
        visibility=torch.ones(1, 1, 2),
        scores=torch.ones(1, 1),
        category=torch.full((1, 1), 2),
        valid_queries=torch.ones(1, 1, dtype=torch.bool),
        nms_keep_queries=torch.ones(1, 1, dtype=torch.bool),
        anchor_y_steps=torch.tensor([3., 4.]),
    )
    batch = dict(
        gt_lanes_3d=torch.zeros(1, 1, 2, 3),
        gt_lanes_visibility=torch.ones(1, 1, 2),
        gt_lanes_category=torch.full((1, 1), 2),
        raw_label=[raw_label],
        img_metas=[{}],
    )

    samples = LaneLoomModel._build_evaluation_samples(predictions, batch)

    assert samples[0]['raw_label'] is raw_label


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
    assert values['num_matches'] == 1.0
    assert values['num_recall_matches'] == 0.0
    assert values['num_precision_matches'] == 1.0
    assert values['F1'] == 0.0
    assert values['Category Accuracy'] == 1.0


def test_anchor_metric_counts_official_recall_and_precision_matches_separately():
    metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, score_threshold=0.3,
        distance_threshold=0.5, match_ratio_threshold=0.75))
    sample = _sample()
    sample['gt_lanes_visibility'][0, :2] = 0.0
    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['num_matches'] == 1.0
    assert values['num_recall_matches'] == 1.0
    assert values['num_precision_matches'] == 0.0
    assert values['Recall'] == 1.0
    assert values['Precision'] == 0.0
    assert values['F1'] == 0.0


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


def test_anchor_metric_can_use_temporal_supplement_but_defaults_to_official_gt():
    sample = _sample()
    enhanced = sample['gt_lanes_3d'].clone()
    enhanced[0, :, 0] += 8.0
    sample['gt_lanes_3d_enhanced'] = enhanced
    sample['gt_lanes_visibility_enhanced'] = sample['gt_lanes_visibility'].clone()

    temporal_metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, distance_threshold=0.5,
        use_temporal_gt_supplement=True))
    temporal_metric.process(None, [sample])
    assert temporal_metric.compute_metrics(temporal_metric.results)['F1'] == 0.0

    current_metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None, distance_threshold=0.5))
    current_metric.process(None, [sample])
    assert current_metric.compute_metrics(current_metric.results)['F1'] == 1.0


def test_anchor_metric_uses_temporal_supplement_category():
    sample = _sample(category=2, predicted_category=7)
    sample['gt_lanes_3d_enhanced'] = sample['gt_lanes_3d'].clone()
    sample['gt_lanes_visibility_enhanced'] = sample['gt_lanes_visibility'].clone()
    sample['gt_lanes_category_enhanced'] = torch.tensor([7, 0])
    metric = METRICS.build(dict(
        type='OpenLaneAnchorMetric', prefix=None,
        use_temporal_gt_supplement=True))

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Category Accuracy'] == 1.0


def test_config_enables_mmengine_validation_and_test_evaluators():
    config_path = Path(
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')
    cfg = Config.fromfile(config_path)
    source = config_path.read_text()

    assert "'./_base_/evaluators/openlane_official.py'" in source
    assert 'dv3dlane_evaluator = dict(' not in source
    assert cfg.val_cfg.type == 'ValLoop'
    assert cfg.test_cfg.type == 'TestLoop'
    assert cfg.val_dataloader is not None
    assert cfg.test_dataloader is not None
    assert cfg.evaluation_protocol in ('dv3dlane', 'official')
    assert cfg.temporal_gt_augmentation is True
    assert cfg.val_evaluator.type == 'OpenLaneOfficialMetric'
    assert cfg.val_evaluator.prefix == 'OpenLaneTemporal'
    assert cfg.val_evaluator.use_temporal_gt_supplement is True
    assert cfg.test_evaluator.type == 'OpenLaneOfficialMetric'
    assert cfg.test_evaluator.prefix == 'OpenLaneOfficial'
    assert cfg.test_evaluator.use_temporal_gt_supplement is False
    assert cfg.test_dataloader.dataset.split == 'test'
    assert cfg.test_dataloader.dataset.dataset_cfg.load_interval == 1
    assert cfg.default_hooks.checkpoint.save_best == 'OpenLaneTemporal/F1'
    assert cfg.default_hooks.checkpoint.rule == 'greater'


def test_config_can_select_dv3dlane_validation_but_keeps_official_test():
    config_path = Path(
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')
    source = config_path.read_text()
    source = source.replace(
        "'./_base_/datasets/openlane_datasets_D5.py'",
        repr(str(config_path.parent / '_base_/datasets/openlane_datasets_D5.py')))
    source = source.replace(
        "'./_base_/evaluators/openlane_official.py'",
        repr(str(
            config_path.parent
            / '_base_/evaluators/openlane_dv3dlane.py')))
    source = source.replace(
        "'./_base_/schedules/schedule_24e_adamw_cosine.py'",
        repr(str(
            config_path.parent
            / '_base_/schedules/schedule_24e_adamw_cosine.py')))

    cfg = Config.fromstring(source, '.py')

    assert cfg.val_evaluator.type == 'OpenLaneAnchorMetric'
    assert cfg.val_evaluator.prefix == 'OpenLaneTemporal'
    assert cfg.val_evaluator.use_temporal_gt_supplement is True
    assert cfg.test_evaluator.type == 'OpenLaneOfficialMetric'
    assert cfg.default_hooks.checkpoint.save_best == 'OpenLaneTemporal/F1'


def test_openlane_test_configs_select_validation_and_each_official_case():
    cfg = Config.fromfile(
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')

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
        assert case_cfg.test_evaluator.prefix == f'OpenLaneOfficial/{name}'

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
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')

    case_configs = build_openlane_test_configs(cfg, selection)

    assert [name for name, _ in case_configs] == expected


def test_openlane_test_configs_reject_unknown_case():
    cfg = Config.fromfile(
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')

    with pytest.raises(ValueError, match='Unknown OpenLane test case'):
        build_openlane_test_configs(cfg, 'unknown')


def test_openlane_test_configs_build_one_pass_score_threshold_scan():
    cfg = Config.fromfile(
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')

    cases = build_openlane_test_configs(
        cfg, 'validation', score_thresholds=[0.8, 0.3, 0.5, 0.5])

    assert len(cases) == 1
    _, scan_cfg = cases[0]
    assert scan_cfg.model.lane_head.test_cfg.score_threshold == 0.3
    assert [metric.score_threshold for metric in scan_cfg.test_evaluator] == [
        0.3, 0.5, 0.8]
    assert [metric.prefix for metric in scan_cfg.test_evaluator] == [
        'OpenLaneOfficial/validation@0.3',
        'OpenLaneOfficial/validation@0.5',
        'OpenLaneOfficial/validation@0.8',
    ]


@pytest.mark.parametrize('thresholds', [[], [0.0], [1.0], [-0.1], [1.1]])
def test_openlane_test_configs_reject_invalid_score_threshold_scan(thresholds):
    cfg = Config.fromfile(
        '/workspace/LaneLoom/configs/laneloom_openlane_image_lidar.py')

    with pytest.raises(ValueError, match='score thresholds'):
        build_openlane_test_configs(
            cfg, 'validation', score_thresholds=thresholds)


def test_logger_reports_validation_progress_and_final_lane_metrics(monkeypatch):
    messages = []
    runner = SimpleNamespace(
        rank=0,
        iter=123,
        val_dataloader=[None] * 100,
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(add_scalars=lambda *_args, **_kwargs: None),
    )
    hook = TrainingLoggerHook(interval=50)
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


def test_training_logger_brief_mode_only_prints_key_lane_losses(monkeypatch):
    messages = []
    scalars = []
    runner = SimpleNamespace(
        rank=0,
        iter=49,
        max_iters=100,
        epoch=0,
        max_epochs=24,
        train_dataloader=[None] * 100,
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(
            add_scalars=lambda values, step: scalars.append((values, step))),
    )
    hook = TrainingLoggerHook(interval=50, mode='brief')
    clock = iter([0.0, 50.0])
    monkeypatch.setattr(logger_module.time, 'monotonic', lambda: next(clock))
    outputs = dict(
        loss=3.3,
        loss_score=0.1,
        loss_category=0.2,
        loss_x=0.3,
        loss_z=0.4,
        loss_visibility=0.5,
        loss_tube=0.55,
        loss_lidar_shape_cover=0.6,
        loss_lidar_shape_mode=0.7,
        loss_lidar_shape_diversity=0.8,
        loss_q0_geometry=0.9,
        loss_aux_0_score=1.0,
        diag_gt_match_recall=1.0,
    )

    hook.before_train(runner)
    hook.after_train_iter(runner, batch_idx=49, outputs=outputs)

    message = messages[0]
    assert 'loss_score:0.1000' in message
    assert 'loss_category:0.2000' in message
    assert 'loss_x:0.3000' in message
    assert 'loss_z:0.4000' in message
    assert 'loss_visibility:0.5000' in message
    assert 'loss_tube:0.5500' in message
    assert 'loss_lidar_shape_cover:0.6000' in message
    assert 'loss_lidar_shape_mode:0.7000' in message
    assert 'loss_lidar_shape_diversity:0.8000' in message
    assert 'loss_q0_geometry:0.9000' in message
    assert 'loss_aux_0_score' not in message
    assert 'diag_gt_match_recall' not in message
    assert 'Loss(avg1) 3.30000000' in message
    assert scalars == [({
        f'train/{name}': value for name, value in outputs.items()
    }, 50)]


def test_training_logger_brief_mode_prints_window_averages(monkeypatch):
    messages = []
    runner = SimpleNamespace(
        rank=0,
        iter=0,
        max_iters=2,
        epoch=0,
        max_epochs=1,
        train_dataloader=[None] * 2,
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(add_scalars=lambda *_args, **_kwargs: None),
    )
    hook = TrainingLoggerHook(
        interval=2, mode='brief', brief_metrics=['loss_x'], window_size=2)
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(logger_module.time, 'monotonic', lambda: next(clock))

    hook.before_train(runner)
    hook.after_train_iter(
        runner, batch_idx=0, outputs=dict(loss=4.0, loss_x=2.0))
    runner.iter = 1
    hook.after_train_iter(
        runner, batch_idx=1, outputs=dict(loss=2.0, loss_x=1.0))

    assert 'Loss(avg2) 3.00000000' in messages[0]
    assert 'loss_x:1.5000' in messages[0]


def test_training_logger_buffers_detached_tensors_without_converting_them():
    runner = SimpleNamespace(rank=0, iter=0)
    hook = TrainingLoggerHook(
        interval=2, mode='brief', brief_metrics=['loss_x'], window_size=2)
    loss = torch.tensor(4.0, requires_grad=True)
    loss_x = torch.tensor(2.0, requires_grad=True)

    hook.after_train_iter(
        runner, batch_idx=0, outputs=dict(loss=loss, loss_x=loss_x))

    buffered_loss = hook.metric_windows['loss'][0]
    buffered_loss_x = hook.metric_windows['loss_x'][0]
    assert isinstance(buffered_loss, torch.Tensor)
    assert isinstance(buffered_loss_x, torch.Tensor)
    assert buffered_loss.requires_grad is False
    assert buffered_loss_x.requires_grad is False


def test_training_logger_defaults_to_detailed_mode(monkeypatch):
    messages = []
    runner = SimpleNamespace(
        rank=0,
        iter=0,
        max_iters=1,
        epoch=0,
        max_epochs=1,
        train_dataloader=[None],
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(add_scalars=lambda *_args, **_kwargs: None),
    )
    hook = TrainingLoggerHook(interval=1)
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(logger_module.time, 'monotonic', lambda: next(clock))

    hook.before_train(runner)
    hook.after_train_iter(
        runner,
        batch_idx=0,
        outputs=dict(
            loss=1.0,
            loss_aux_0_objectness=0.25,
            diag_gt_match_recall=0.75))

    assert 'loss_aux_0_objectness:0.2500' in messages[0]
    assert 'diag_gt_match_recall:0.7500' in messages[0]


def test_training_logger_rejects_string_brief_metrics():
    with pytest.raises(TypeError, match='sequence of metric names'):
        TrainingLoggerHook(mode='brief', brief_metrics='loss_x')


@pytest.mark.parametrize('window_size', [0, -1, 2.5, True])
def test_training_logger_rejects_invalid_window_size(window_size):
    with pytest.raises(ValueError, match='positive integer'):
        TrainingLoggerHook(mode='brief', window_size=window_size)


def test_logger_reports_test_case_progress_eta_and_final_batch(monkeypatch):
    messages = []
    runner = SimpleNamespace(
        rank=0,
        work_dir='/tmp/nms_evaluation/curve_case/logs',
        test_dataloader=[None] * 320,
        logger=SimpleNamespace(info=messages.append),
    )
    hook = TrainingLoggerHook(interval=50)
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


def test_logger_reports_complete_test_case_metrics():
    messages = []
    scalars = []
    runner = SimpleNamespace(
        rank=0,
        iter=0,
        logger=SimpleNamespace(info=messages.append),
        visualizer=SimpleNamespace(
            add_scalars=lambda values, step: scalars.append((values, step))),
    )
    metrics = {
        'OpenLaneOfficial/curve_case/F1': 0.70423734,
        'OpenLaneOfficial/curve_case/Recall': 0.65300149,
        'OpenLaneOfficial/curve_case/Precision': 0.76419789,
        'OpenLaneOfficial/curve_case/Category Accuracy': 0.95413834,
        'OpenLaneOfficial/curve_case/num_gt': 32784.0,
        'OpenLaneOfficial/curve_case/num_pred': 25127.0,
        'OpenLaneOfficial/curve_case/num_tp': 24116.0,
        'OpenLaneOfficial/curve_case/num_pred_before_nms': 180000.0,
        'OpenLaneOfficial/curve_case/num_nms_suppressed': 154873.0,
        'OpenLaneOfficial/curve_case/nms_suppression_ratio': 0.86040556,
        'OpenLaneOfficial/curve_case/x_error_near': 0.12943050,
        'OpenLaneOfficial/curve_case/x_error_far': 0.16741155,
        'OpenLaneOfficial/curve_case/z_error_near': 0.03810635,
        'OpenLaneOfficial/curve_case/z_error_far': 0.06069867,
    }

    TrainingLoggerHook().after_test_epoch(runner, metrics)

    assert len(messages) == 1
    summary = messages[0]
    assert 'Evaluation laneline F1: 0.70423734' in summary
    assert 'Recall: 0.65300149' in summary
    assert 'Precision: 0.76419789' in summary
    assert 'Category Accuracy: 0.95413834' in summary
    assert 'GT / Pred / TP: 32784 / 25127 / 24116' in summary
    assert 'x error (close): 0.12943050 m' in summary
    assert scalars == [({
        f'test/{name}': value for name, value in metrics.items()
    }, 0)]


def test_train_experiment_layout_uses_a_dated_laneloom_directory(tmp_path):
    cfg = Config(dict(
        work_dir=str(tmp_path / 'work_dirs' / 'laneloom'),
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=1))))
    experiment_dir = configure_train_experiment(
        cfg, now=datetime(2026, 7, 12, 18, 30, 45))

    assert experiment_dir.name == 'exp_date_20260712_183045'
    assert (experiment_dir / 'logs').is_dir()
    assert (experiment_dir / 'Tensorboard').is_dir()
    assert (experiment_dir / 'checkpoints').is_dir()
    assert cfg.work_dir == str(experiment_dir / 'logs')
    assert cfg.default_hooks.checkpoint.type == 'ExperimentCheckpointHook'
    assert cfg.default_hooks.checkpoint.out_dir == str(experiment_dir / 'checkpoints')
    assert [backend.type for backend in cfg.visualizer.vis_backends] == [
        'TensorboardVisBackend']


def test_laneloom_runner_writes_log_directly_in_work_dir(tmp_path):
    logs_dir = tmp_path / 'logs'
    runner = ExperimentRunner(
        model=torch.nn.Linear(1, 1),
        work_dir=str(logs_dir),
        default_scope=None,
        experiment_name='direct_log_success',
    )

    assert runner.log_dir == str(logs_dir)
    assert len(list(logs_dir.glob('*.log'))) == 1
    assert not [path for path in logs_dir.iterdir() if path.is_dir()]


def test_laneloom_runner_does_not_create_log_subdir_when_init_fails(tmp_path):
    logs_dir = tmp_path / 'logs'

    with pytest.raises(KeyError):
        ExperimentRunner(
            model=dict(type='MissingLaneLoomModel'),
            work_dir=str(logs_dir),
            default_scope=None,
            experiment_name='direct_log_failure',
        )

    assert not [path for path in logs_dir.iterdir() if path.is_dir()]
