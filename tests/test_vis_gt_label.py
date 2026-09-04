import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from LaneLoom.tests.test_vis_input import (
    _build_once_sample,
    _build_openlane_sample,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ONCE_SCRIPT = REPO_ROOT / 'vis/gt_label/visualize_once_gt.py'
OPENLANE_SCRIPT = REPO_ROOT / 'vis/gt_label/visualize_openlane_gt.py'


def _run_script(script_path, image_path, output_root, data_root):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT.parent)
    env['LANELOOM_DATA_ROOT'] = str(data_root)
    return subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(image_path),
            '--output-root',
            str(output_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_once_gt_cli_groups_overlay_and_bev_by_dataset_scene_sample(tmp_path):
    image_path = _build_once_sample(tmp_path)
    output_root = tmp_path / 'gt_label'

    completed = _run_script(
        ONCE_SCRIPT, image_path, output_root, data_root=tmp_path)

    assert completed.returncode == 0, completed.stderr
    sample_dir = output_root / 'once/000001/100'
    overlay_path = sample_dir / '01_gt_overlay.jpg'
    bev_path = sample_dir / '02_gt_bev.jpg'
    assert overlay_path.is_file()
    assert bev_path.is_file()
    assert not np.array_equal(
        np.asarray(Image.open(image_path)),
        np.asarray(Image.open(overlay_path)),
    )
    assert Image.open(bev_path).size == (800, 1000)


def test_openlane_gt_cli_groups_overlay_and_bev_by_dataset_scene_sample(
        tmp_path):
    image_path, scene_name, frame_key = _build_openlane_sample(tmp_path)
    output_root = tmp_path / 'gt_label'

    completed = _run_script(
        OPENLANE_SCRIPT, image_path, output_root, data_root=tmp_path)

    assert completed.returncode == 0, completed.stderr
    sample_dir = output_root / 'openlane' / scene_name / frame_key
    overlay_path = sample_dir / '01_gt_overlay.jpg'
    bev_path = sample_dir / '02_gt_bev.jpg'
    assert overlay_path.is_file()
    assert bev_path.is_file()
    assert not np.array_equal(
        np.asarray(Image.open(image_path)),
        np.asarray(Image.open(overlay_path)),
    )
    assert Image.open(bev_path).size == (800, 1000)
