import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).with_name('infer_openlane_sample.py')
SPEC = importlib.util.spec_from_file_location('infer_openlane_sample', SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _lane():
    return MODULE.LanePrediction(
        xyz=np.asarray([[0.0, 1.0, 0.0], [1.0, 2.0, 0.1]]),
        score=0.9,
    )


def test_bev_output_uses_concise_axis_labels_without_title(tmp_path):
    with patch.object(MODULE.plt, 'close'):
        MODULE._save_bev([_lane()], tmp_path / '03_bev.png')
        figure = MODULE.plt.gcf()

    axis = figure.axes[0]
    assert axis.get_xlabel() == 'x / m'
    assert axis.get_ylabel() == 'y / m'
    assert axis.get_title() == ''
    MODULE.plt.close(figure)


def test_3d_output_has_no_title(tmp_path):
    with patch.object(MODULE.plt, 'close'):
        MODULE._save_3d([_lane()], tmp_path / '04_3d.png')
        figure = MODULE.plt.gcf()

    axis = figure.axes[0]
    assert axis.get_title() == ''
    MODULE.plt.close(figure)
