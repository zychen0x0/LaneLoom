"""CPU checks for the E6 shape-mode statistic with the real prior class."""

import importlib
import sys

import torch

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from test_modules import _install_stand_ins  # noqa: E402

_install_stand_ins()
e6 = importlib.import_module('LaneLoom.ablation_extras.tools.e6_shape_modes')
prior_module = importlib.import_module(
    'LaneLoom.models.head.lidar_shape_token_prior')


def _lane(x0, bend=0.0):
    y = torch.arange(2.0, 101.0, 2.0)
    x = x0 + bend * (y / 100.0) ** 2
    return torch.stack((x, y, torch.zeros_like(y)), dim=-1)


def test_parallel_lanes_share_one_mode_and_curve_adds_one():
    prior = prior_module.LiDARShapeTokenPrior(
        lidar_in_channels=8, anchor_y_steps=[2.0 * i for i in range(1, 51)],
        max_templates=5)
    lanes = torch.zeros(20, 50, 3)
    visibility = torch.zeros(20, 50)
    for index, (x0, bend) in enumerate(((-5.5, 0), (-1.8, 0), (1.8, 0),
                                        (5.5, 0), (9.0, 12.0))):
        lanes[index] = _lane(x0, bend)
        visibility[index] = 1
    modes, eligible = e6.frame_modes(prior, lanes, visibility)
    assert eligible == 5
    assert modes == 2
    assert e6.frame_modes(prior, torch.zeros(20, 50, 3),
                          torch.zeros(20, 50)) == (0, 0)


def test_summary_cumulative_share():
    summary = e6.summarize([1, 1, 2, 3, 0], [4, 4, 5, 6, 0])
    assert summary['frames_with_lanes'] == 4
    assert summary['cumulative_percent']['<= 1'] == 50.0
    assert summary['cumulative_percent']['<= 3'] == 100.0
