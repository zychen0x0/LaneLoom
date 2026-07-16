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


def build_openlane_test_configs(cfg, case='all'):
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

    case_configs = []
    for name, split in selected:
        case_cfg = deepcopy(cfg)
        case_cfg.test_dataloader.dataset.split = 'val'
        case_cfg.test_dataloader.dataset.dataset_cfg.val_split = split
        case_cfg.test_evaluator.prefix = f'OpenLane/{name}'
        case_configs.append((name, case_cfg))
    return case_configs
