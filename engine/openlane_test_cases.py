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


def score_threshold_metric_prefix(case_name, threshold):
    return f'OpenLane/{case_name}@{threshold:g}'


def build_openlane_test_configs(cfg, case='all', score_thresholds=None):
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
    case_configs = []
    for name, split in selected:
        case_cfg = deepcopy(cfg)
        case_cfg.test_dataloader.dataset.split = 'val'
        case_cfg.test_dataloader.dataset.dataset_cfg.val_split = split
        case_cfg.test_evaluator.prefix = f'OpenLane/{name}'
        if score_thresholds is not None:
            base_evaluator = case_cfg.test_evaluator
            case_cfg.model.lane_head.test_cfg.score_threshold = min(
                score_thresholds)
            case_cfg.test_evaluator = []
            for threshold in score_thresholds:
                evaluator = deepcopy(base_evaluator)
                evaluator.score_threshold = threshold
                evaluator.prefix = score_threshold_metric_prefix(
                    name, threshold)
                case_cfg.test_evaluator.append(evaluator)
        case_configs.append((name, case_cfg))
    return case_configs
