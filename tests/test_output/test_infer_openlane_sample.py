import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / 'vis/output/infer_openlane_sample.py'
SPEC = importlib.util.spec_from_file_location('infer_openlane_sample', SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _lane():
    return MODULE.LanePrediction(
        xyz=np.asarray([[0.0, 1.0, 0.0], [1.0, 2.0, 0.1]]),
        score=0.9,
    )


def _assert_png_and_svg_exports(png_path):
    svg_path = png_path.with_suffix('.svg')
    pdf_path = png_path.with_suffix('.pdf')
    assert png_path.is_file()
    assert svg_path.is_file()
    assert not pdf_path.exists()

    with Image.open(png_path) as image:
        dpi_x, dpi_y = image.info['dpi']
        assert dpi_x >= 599
        assert dpi_y >= 599

    svg = svg_path.read_text(encoding='utf-8')
    assert '<svg' in svg
    assert '<text' in svg


def test_sample_output_dir_groups_by_dataset_scene_and_frame(tmp_path):
    image_path = (
        tmp_path / 'images/validation/segment-example/1616003692000.jpg')

    assert MODULE._sample_output_dir(image_path, tmp_path / 'output') == (
        tmp_path / 'output/openlane/segment-example/1616003692000')


def test_legacy_output_dir_argument_is_still_supported(tmp_path):
    output_dir = tmp_path / 'legacy-output'
    with patch.object(
            sys, 'argv', ['infer_openlane_sample.py', '--output-dir',
                          str(output_dir)]):
        args = MODULE.parse_args()

    assert args.output_dir == output_dir


def test_bev_output_uses_concise_axis_labels_without_title(tmp_path):
    output_path = tmp_path / '03_bev.png'
    with patch.object(MODULE.plt, 'close'):
        MODULE._save_bev([_lane()], output_path)
        figure = MODULE.plt.gcf()

    axis = figure.axes[0]
    assert axis.get_xlabel() == 'x / m'
    assert axis.get_ylabel() == 'y / m'
    assert axis.get_title() == ''
    _assert_png_and_svg_exports(output_path)
    MODULE.plt.close(figure)


def test_3d_output_has_no_title(tmp_path):
    output_path = tmp_path / '04_3d.png'
    with patch.object(MODULE.plt, 'close'):
        MODULE._save_3d([_lane()], output_path)
        figure = MODULE.plt.gcf()

    axis = figure.axes[0]
    assert axis.get_title() == ''
    _assert_png_and_svg_exports(output_path)
    MODULE.plt.close(figure)
