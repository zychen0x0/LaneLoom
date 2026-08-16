import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / 'vis/input/visualize_sample.py'


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _build_once_sample(data_root):
    once_root = data_root / 'once_3dlanes'
    scene_name = '000001'
    frame_key = '100'

    image_path = (
        once_root / '3D_images/data' / scene_name / 'cam01'
        / f'{frame_key}.jpg')
    image_path.parent.mkdir(parents=True)
    Image.new('RGB', (192, 102), color=(30, 60, 90)).save(image_path)

    label = {
        'lane_num': 1,
        'lanes': [[[1.0, 2.0, 5.0], [1.0, 2.0, 10.0]]],
        'calibration': [
            [100.0, 0.0, 96.0, 0.0],
            [0.0, 100.0, 51.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    }
    _write_json(
        once_root / 'ONCE_3DLanes/val' / scene_name / 'cam01'
        / f'{frame_key}.json',
        label)

    lidar_path = (
        once_root / '3D_lidars/data' / scene_name / 'lidar_roof'
        / f'{frame_key}.bin')
    lidar_path.parent.mkdir(parents=True)
    np.asarray([[1.0, 2.0, 5.0, 0.5]], dtype=np.float32).tofile(lidar_path)

    _write_json(
        once_root / '3D_infos/data' / scene_name / f'{scene_name}.json',
        {
            'calib': {
                'cam01': {'cam_to_velo': np.eye(4).reshape(-1).tolist()},
            },
            'frames': [{
                'frame_id': frame_key,
                'pose': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }],
        })
    return image_path


def _build_openlane_sample(data_root):
    scene_name = 'segment-test_with_camera_labels'
    timestamp = 123456789012345600
    frame_key = str(timestamp)[:12]
    image_path = (
        data_root / 'openlane/images/training' / scene_name
        / f'{timestamp}.jpg')
    image_path.parent.mkdir(parents=True)
    Image.new('RGB', (192, 128), color=(30, 60, 90)).save(image_path)

    _write_json(
        data_root / 'openlane/lane3d_1000/training' / scene_name
        / f'{timestamp}.json',
        {
            'intrinsic': [
                [100.0, 0.0, 96.0],
                [0.0, 100.0, 64.0],
                [0.0, 0.0, 1.0],
            ],
            'extrinsic': np.eye(4).tolist(),
            'pose': np.eye(4).tolist(),
            'lane_lines': [{
                'xyz': [[5.0, 10.0], [1.0, 1.0], [-2.0, -2.0]],
                'visibility': [1.0, 1.0],
                'category': 1,
            }],
        })

    lidar_dir = data_root / 'waymo/waymo_processed_data' / scene_name
    lidar_dir.mkdir(parents=True)
    np.save(
        lidar_dir / '0000.npy',
        np.asarray([[1.0, 2.0, 3.0, 0.5, 0.25, -1.0]], dtype=np.float32))
    with open(lidar_dir / f'{scene_name}.pkl', 'wb') as handle:
        pickle.dump([{
            'metadata': {'timestamp_micros': timestamp},
            'point_cloud': {'sample_idx': 0},
            'pose': np.eye(4, dtype=np.float32),
        }], handle)
    return image_path, scene_name, frame_key


def test_cli_visualizes_once_image_gt_augmentations_and_point_cloud(tmp_path):
    image_path = _build_once_sample(tmp_path)
    output_root = tmp_path / 'visualizations'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT.parent)
    env['LANELOOM_DATA_ROOT'] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(image_path),
            '--dataset',
            'once',
            '--output-root',
            str(output_root),
            '--seed',
            '7',
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    sample_dir = output_root / 'once/000001/100'
    expected_files = [
        sample_dir / '01_original.jpg',
        sample_dir / '02_gt_overlay.jpg',
        sample_dir / '03_augmentations.jpg',
        sample_dir / '04_points.ply',
    ]
    assert all(path.is_file() for path in expected_files)
    assert Image.open(expected_files[0]).size == (192, 102)
    assert not np.array_equal(
        np.asarray(Image.open(expected_files[0])),
        np.asarray(Image.open(expected_files[1])),
    )
    ply_header = expected_files[3].read_bytes()[:256]
    assert b'element vertex 1' in ply_header
    ply_bytes = expected_files[3].read_bytes()
    vertex_offset = ply_bytes.index(b'end_header\n') + len(b'end_header\n')
    vertex_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<f4'), ('elongation', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex = np.frombuffer(
        ply_bytes, dtype=vertex_dtype, count=1, offset=vertex_offset)[0]
    np.testing.assert_allclose(
        [vertex['x'], vertex['y'], vertex['z']],
        [1.0, 5.0, -2.0],
        atol=1e-6)


def test_cli_auto_detects_and_visualizes_openlane_sample(tmp_path):
    image_path, scene_name, frame_key = _build_openlane_sample(tmp_path)
    output_root = tmp_path / 'visualizations'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT.parent)
    env['LANELOOM_DATA_ROOT'] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(image_path),
            '--output-root',
            str(output_root),
            '--seed',
            '7',
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    sample_dir = output_root / 'openlane' / scene_name / frame_key
    original_path = sample_dir / '01_original.jpg'
    overlay_path = sample_dir / '02_gt_overlay.jpg'
    assert original_path.is_file()
    assert overlay_path.is_file()
    assert not np.array_equal(
        np.asarray(Image.open(original_path)),
        np.asarray(Image.open(overlay_path)),
    )
    assert (sample_dir / '03_augmentations.jpg').is_file()
    points_path = sample_dir / '04_points.ply'
    assert points_path.is_file()
    ply_bytes = points_path.read_bytes()
    vertex_offset = ply_bytes.index(b'end_header\n') + len(b'end_header\n')
    vertex_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('intensity', '<f4'), ('elongation', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex = np.frombuffer(
        ply_bytes, dtype=vertex_dtype, count=1, offset=vertex_offset)[0]
    np.testing.assert_allclose(
        [vertex['x'], vertex['y'], vertex['z']],
        [-2.0, 1.0, 3.0],
        atol=1e-6)
