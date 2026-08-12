import json

import numpy as np

import LaneLoom  # noqa: F401 - register project metrics.
from mmdet3d.registry import METRICS


def _sample(include_false_positive=False):
    pred_x = [[-1.0, -1.0, -1.0]]
    pred_z = [[-1.0, -1.0, -1.0]]
    scores = [0.9]
    visibility = [[1.0, 1.0, 1.0]]
    if include_false_positive:
        pred_x.append([5.0, 5.0, 5.0])
        pred_z.append([-1.0, -1.1, -1.2])
        scores.append(0.8)
        visibility.append([1.0, 1.0, 1.0])
    return dict(
        pred_x=np.asarray(pred_x, dtype=np.float32),
        pred_z=np.asarray(pred_z, dtype=np.float32),
        pred_visibility=np.asarray(visibility, dtype=np.float32),
        pred_scores=np.asarray(scores, dtype=np.float32),
        pred_category=np.zeros(len(scores), dtype=np.int64),
        pred_nms_keep_queries=np.ones(len(scores), dtype=bool),
        anchor_y_steps=np.asarray([1.0, 5.0, 9.0], dtype=np.float32),
        scene_name='000001',
        frame_key='100',
        raw_label={
            'lane_num': 1,
            'lanes': [[[-1.0, 1.0, 1.0],
                       [-1.0, 1.0, 5.0],
                       [-1.0, 1.0, 9.0]]],
        },
    )


def test_once_official_metric_reports_perfect_camera_coordinate_prediction():
    metric = METRICS.build(dict(
        type='ONCEOfficialMetric', prefix=None, score_threshold=0.5,
        score_thresholds=[0.5, 0.85]))
    metric.process(None, [_sample()])
    values = metric.compute_metrics(metric.results)

    assert values['F1'] == 1.0
    assert values['Precision'] == 1.0
    assert values['Recall'] == 1.0
    assert values['CDE'] == 0.0
    assert values['best_F1'] == 1.0
    assert values['best_score_threshold'] == 0.5


def test_once_official_metric_scans_scores_without_hiding_primary_threshold():
    metric = METRICS.build(dict(
        type='ONCEOfficialMetric', prefix=None, score_threshold=0.5,
        score_thresholds=[0.5, 0.85]))
    metric.process(None, [_sample(include_false_positive=True)])
    values = metric.compute_metrics(metric.results)

    np.testing.assert_allclose(values['Precision'], 0.5)
    np.testing.assert_allclose(values['Recall'], 1.0)
    np.testing.assert_allclose(values['F1'], 2.0 / 3.0)
    assert values['best_F1'] == 1.0
    assert values['best_score_threshold'] == 0.85


def test_once_official_metric_exports_benchmark_json(tmp_path):
    metric = METRICS.build(dict(
        type='ONCEOfficialMetric', prefix=None, score_threshold=0.5,
        prediction_dir=str(tmp_path)))
    metric.process(None, [_sample()])
    metric.compute_metrics(metric.results)

    output = tmp_path / '000001/cam01/100.json'
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['lanes'][0]['score'] == 0.9
    np.testing.assert_allclose(
        payload['lanes'][0]['points'],
        [[-1.0, 1.0, 1.0], [-1.0, 1.0, 5.0], [-1.0, 1.0, 9.0]])
