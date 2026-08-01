"""Official OpenLane split selection for one-command checkpoint testing."""
from __future__ import annotations

from copy import deepcopy


OPENLANE_TEST_SPLITS = (
    ('validation', 'validation'),
    ('up_down_case', 'test/up_down_case'),
    ('curve_case', 'test/curve_case'),
    ('extreme_weather_case', 'test/extreme_weather_case'),
    ('intersection_case', 'test/intersection_case'),
    ('merge_split_case', 'test/merge_split_case'),
    ('night_case', 'test/night_case'),
)
OPENLANE_TEST_CASE_CHOICES = (
    'all', 'validation', 'test',
    *(name for name, _ in OPENLANE_TEST_SPLITS if name != 'validation'),
)


def normalise_score_thresholds(score_thresholds):
    if score_thresholds is None:
        return None
    thresholds = sorted(set(float(value) for value in score_thresholds))
    if not thresholds or any(
            not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError('score thresholds must contain values in (0, 1).')
    return thresholds


def normalise_distance_nms_thresholds(distance_nms_thresholds):
    if distance_nms_thresholds is None:
        return None
    thresholds = sorted(set(
        float(value) for value in distance_nms_thresholds))
    if not thresholds or any(threshold <= 0.0 for threshold in thresholds):
        raise ValueError(
            'distance NMS thresholds must contain positive values.')
    return thresholds


def score_threshold_metric_prefix(
        case_name, threshold, metric_prefix='OpenLaneOfficial'):
    return f'{metric_prefix}/{case_name}@{threshold:g}'


def distance_nms_metric_prefix(
        case_name, threshold, metric_prefix='OpenLaneOfficial'):
    return f'{metric_prefix}/{case_name}@distance={threshold:g}'


def select_recall_safe_distance_threshold(case_results):
    """Maximise validation F1 without lowering recall on any case."""
    if not case_results:
        return None
    common_thresholds = None
    for results in case_results.values():
        thresholds = set(results) - {None}
        common_thresholds = (
            thresholds if common_thresholds is None
            else common_thresholds & thresholds)
    eligible = [
        threshold for threshold in (common_thresholds or ())
        if all(
            results[threshold]['Recall'] + 1e-12
            >= results[None]['Recall']
            for results in case_results.values())
    ]
    if not eligible:
        return None
    selection_case = (
        case_results['validation']
        if 'validation' in case_results
        else next(iter(case_results.values())))
    return max(eligible, key=lambda threshold: (
        selection_case[threshold]['F1'],
        selection_case[threshold]['Precision'],
        -threshold))


def build_openlane_test_configs(
        cfg, case='all', score_thresholds=None,
        distance_nms_thresholds=None):
    """Return independent test configs for validation and official case splits."""
    if case not in OPENLANE_TEST_CASE_CHOICES:
        raise ValueError(
            f'Unknown OpenLane test case {case!r}; expected one of '
            f'{OPENLANE_TEST_CASE_CHOICES}.')
    selected = OPENLANE_TEST_SPLITS
    if case == 'validation':
        selected = OPENLANE_TEST_SPLITS[:1]
    elif case == 'test':
        selected = OPENLANE_TEST_SPLITS[1:]
    elif case != 'all':
        selected = tuple(
            item for item in OPENLANE_TEST_SPLITS if item[0] == case)

    score_thresholds = normalise_score_thresholds(score_thresholds)
    distance_nms_thresholds = normalise_distance_nms_thresholds(
        distance_nms_thresholds)
    if score_thresholds is not None and distance_nms_thresholds is not None:
        raise ValueError(
            'score and distance NMS threshold scans cannot be combined.')
    case_configs = []
    for name, split in selected:
        case_cfg = deepcopy(cfg)
        case_cfg.test_dataloader.dataset.split = 'val'
        case_cfg.test_dataloader.dataset.dataset_cfg.val_split = split
        metric_prefix = case_cfg.test_evaluator.prefix
        case_cfg.test_evaluator.prefix = f'{metric_prefix}/{name}'
        if score_thresholds is not None:
            base_evaluator = case_cfg.test_evaluator
            case_cfg.model.lane_head.test_cfg.score_threshold = min(
                score_thresholds)
            case_cfg.test_evaluator = []
            for threshold in score_thresholds:
                evaluator = deepcopy(base_evaluator)
                evaluator.score_threshold = threshold
                evaluator.prefix = score_threshold_metric_prefix(
                    name, threshold, metric_prefix)
                case_cfg.test_evaluator.append(evaluator)
        elif distance_nms_thresholds is not None:
            base_evaluator = case_cfg.test_evaluator
            case_cfg.test_evaluator = [deepcopy(base_evaluator)]
            for threshold in distance_nms_thresholds:
                evaluator = deepcopy(base_evaluator)
                evaluator.mean_distance_nms_threshold = threshold
                evaluator.prefix = distance_nms_metric_prefix(
                    name, threshold, metric_prefix)
                case_cfg.test_evaluator.append(evaluator)
        case_configs.append((name, case_cfg))
    return case_configs
