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


def build_openlane_test_configs(cfg):
    """Return independent test configs for validation and official case splits."""
    case_configs = []
    for name, split in OPENLANE_TEST_SPLITS:
        case_cfg = deepcopy(cfg)
        case_cfg.test_dataloader.dataset.split = 'val'
        case_cfg.test_dataloader.dataset.dataset_cfg.val_split = split
        case_cfg.test_evaluator.prefix = f'OpenLane/{name}'
        case_configs.append((name, case_cfg))
    return case_configs
