from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine.config import Config, ConfigDict
from PIL import Image

from LaneFormer.tools.inspect_ground_geometry import (
    resolve_image_sample,
    load_points,
    save_outputs,
)


def test_scene_outputs_visual_original_image_and_lidar_ply_only(tmp_path):
    image_path = tmp_path / 'source.jpg'
    Image.new('RGB', (8, 6), color=(20, 40, 60)).save(image_path)
    lidar_path = tmp_path / 'source.npy'
    np.save(lidar_path, np.array(
        [[5., 2., 0., .5, .1, 0.]], dtype=np.float32))
    prefix = tmp_path / 'scene'
    points = torch.tensor([[-2., 5., 0.]])
    result = dict(
        road_valid=torch.tensor([[False]]),
        ground_masks=[torch.tensor([True])],
        road_samples=torch.zeros(1, 1, 3, 3),
        horizontal_curvature=torch.zeros(1, 1, 3),
        vertical_curvature=torch.zeros(1, 1, 3),
        anchor_y_steps=torch.tensor([2., 4., 6.]))
    source = dict(
        image_path=str(image_path),
        lidar_path=str(lidar_path),
        lidar_frame='waymo_vehicle')

    outputs = save_outputs(prefix, points, result, source)

    assert outputs == {
        'visualization': prefix.with_suffix('.png'),
        'original_image': tmp_path / 'scene_image.jpg',
        'original_lidar_current_ground': (
            tmp_path / 'scene_lidar_current_ground.ply'),
    }
    assert outputs['visualization'].is_file()
    assert outputs['original_image'].read_bytes() == image_path.read_bytes()
    ply_text = outputs['original_lidar_current_ground'].read_text()
    assert 'content current_frame_unsegmented_valid_returns' in ply_text
    assert 'coordinates openlane_ground_x_lateral_y_forward_z_up' in ply_text
    assert '-2.000000 5.000000 0.000000' in ply_text
    assert not prefix.with_suffix('.json').exists()
    assert not prefix.with_suffix('.npz').exists()


def test_image_path_resolves_its_openlane_label_and_split(tmp_path):
    image_root = tmp_path / 'images'
    label_root = tmp_path / 'lane3d_1000'
    relative = 'training/segment-example/123456789012345678.jpg'
    image_path = image_root / relative
    label_path = label_root / relative.replace('.jpg', '.json')
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    image_path.touch()
    label_path.touch()
    cfg = ConfigDict(dataset_cfg=dict(
        data_image_dir=str(image_root),
        data_label_dir=str(label_root),
        train_split='training', val_split='validation', test_split='test'))

    resolved_label, loader_split = resolve_image_sample(cfg, image_path)

    assert resolved_label == label_path.resolve()
    assert loader_split == 'train'


OPENLANE_SAMPLE_IMAGE = Path(
    '/workspace/lanedet3d/data/openlane/images/training/'
    'segment-18441113814326864765_725_000_745_000_with_camera_labels/'
    '152284979779166500.jpg')


@pytest.mark.skipif(
    not OPENLANE_SAMPLE_IMAGE.is_file(),
    reason='The requested OpenLane sample is not available.')
def test_image_selection_loads_the_corresponding_scene_and_lidar_queue():
    config_path = Path(__file__).resolve().parents[2] / 'configs' / (
        'laneformer_openlane_image_lidar.py')
    cfg = Config.fromfile(config_path)
    args = SimpleNamespace(
        lidar_file=None,
        lidar_frame='waymo_vehicle',
        image_path=str(OPENLANE_SAMPLE_IMAGE),
        index=None,
        split='train')

    points, source = load_points(cfg, args)

    assert points.numel() > 0
    assert Path(source['image_path']).resolve() == OPENLANE_SAMPLE_IMAGE
    assert Path(source['lidar_path']).parent.name == (
        'segment-18441113814326864765_725_000_745_000_with_camera_labels')
    assert source['frame_key'] == '152284979779'
