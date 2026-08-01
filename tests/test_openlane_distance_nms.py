import numpy as np
import pytest
import torch
from mmengine.config import Config

import LaneLoom  # noqa: F401 - register project metrics.
from LaneLoom.engine.openlane_test_cases import (
    build_openlane_test_configs,
    select_recall_safe_distance_threshold,
)
from mmdet3d.registry import METRICS


def _test_config():
    return Config(dict(
        model=dict(lane_head=dict(test_cfg=dict(score_threshold=0.5))),
        test_dataloader=dict(dataset=dict(
            split='test', dataset_cfg=dict(val_split='validation'))),
        test_evaluator=dict(
            type='OpenLaneOfficialMetric', prefix='OpenLaneOfficial',
            score_threshold=0.5, visibility_threshold=0.5)))


def test_openlane_test_configs_build_one_pass_distance_nms_scan():
    cases = build_openlane_test_configs(
        _test_config(), 'validation',
        distance_nms_thresholds=[0.35, 0.20, 0.30, 0.25, 0.25])

    _, scan_cfg = cases[0]
    assert [
        metric.get('mean_distance_nms_threshold')
        for metric in scan_cfg.test_evaluator
    ] == [None, 0.2, 0.25, 0.3, 0.35]
    assert [metric.prefix for metric in scan_cfg.test_evaluator] == [
        'OpenLaneOfficial/validation',
        'OpenLaneOfficial/validation@distance=0.2',
        'OpenLaneOfficial/validation@distance=0.25',
        'OpenLaneOfficial/validation@distance=0.3',
        'OpenLaneOfficial/validation@distance=0.35',
    ]


def test_distance_nms_scan_forces_a_tube_only_baseline():
    cfg = _test_config()
    cfg.model.lane_head.test_cfg.curve_nms = dict(
        tube_radius=0.5, mean_distance_threshold=0.35)
    cfg.test_evaluator.mean_distance_nms_threshold = 0.35

    [(_, scan_cfg)] = build_openlane_test_configs(
        cfg, 'validation', distance_nms_thresholds=[0.2])

    assert 'mean_distance_threshold' not in (
        scan_cfg.model.lane_head.test_cfg.curve_nms)
    assert scan_cfg.test_evaluator[0].mean_distance_nms_threshold is None


def test_official_metric_applies_distance_nms_after_tube_mask():
    anchors = torch.tensor([3.0, 20.0, 40.0, 60.0])
    pred_x = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1],
        [3.0, 3.0, 3.0, 3.0],
    ])
    sample = dict(
        pred_x=pred_x,
        pred_z=torch.zeros_like(pred_x),
        pred_visibility=torch.ones_like(pred_x),
        pred_scores=torch.tensor([0.9, 0.8, 0.95, 0.7]),
        pred_category=torch.tensor([2, 2, 2, 2]),
        pred_nms_keep_queries=torch.tensor([True, True, False, True]),
        anchor_y_steps=anchors,
        raw_label=dict(
            extrinsic=np.eye(4).tolist(),
            intrinsic=np.eye(3).tolist(), lane_lines=[]),
    )
    metric = METRICS.build(dict(
        type='OpenLaneOfficialMetric', prefix=None, score_threshold=0.5,
        visibility_threshold=0.5, mean_distance_nms_threshold=0.25))

    metric.process(None, [sample])
    values = metric.compute_metrics(metric.results)

    assert values['num_pred'] == 2.0
    assert values['pred_per_frame'] == 2.0


@pytest.mark.parametrize(
    'threshold', [0.0, float('nan'), float('inf')])
def test_official_metric_rejects_non_finite_distance_threshold(threshold):
    with pytest.raises(ValueError, match='finite positive'):
        METRICS.build(dict(
            type='OpenLaneOfficialMetric', prefix=None,
            mean_distance_nms_threshold=threshold))


def test_recall_safe_selection_requires_every_case_to_match_baseline_recall():
    case_results = {
        'validation': {
            None: dict(Recall=0.60, F1=0.50, Precision=0.43),
            0.2: dict(Recall=0.61, F1=0.52, Precision=0.45),
            0.25: dict(Recall=0.60, F1=0.54, Precision=0.49),
            0.3: dict(Recall=0.59, F1=0.55, Precision=0.52),
        },
        'curve_case': {
            None: dict(Recall=0.55, F1=0.48, Precision=0.43),
            0.2: dict(Recall=0.56, F1=0.49, Precision=0.44),
            0.25: dict(Recall=0.55, F1=0.51, Precision=0.48),
            0.3: dict(Recall=0.56, F1=0.52, Precision=0.49),
        },
    }

    selected = select_recall_safe_distance_threshold(case_results)

    assert selected == 0.25


@pytest.mark.parametrize(
    'thresholds', [[], [0.0], [-0.1], [float('nan')], [float('inf')]])
def test_openlane_test_configs_reject_invalid_distance_nms_scan(thresholds):
    with pytest.raises(ValueError, match='distance NMS thresholds'):
        build_openlane_test_configs(
            _test_config(), 'validation',
            distance_nms_thresholds=thresholds)
