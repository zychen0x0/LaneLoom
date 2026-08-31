import json
from pathlib import Path

import numpy as np
import pytest
from mmengine.config import Config
from PIL import Image

import LaneLoom.datasets  # noqa: F401 - register project datasets.
from mmdet3d.registry import DATASETS


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _build_once_fixture(tmp_path: Path):
    root = tmp_path / 'once_3dlanes'
    sequence = '000001'
    frame = '100'

    image_path = root / '3D_images/data' / sequence / 'cam01' / f'{frame}.jpg'
    image_path.parent.mkdir(parents=True)
    Image.new('RGB', (192, 102), color=(30, 60, 90)).save(image_path)

    lidar_path = root / '3D_lidars/data' / sequence / 'lidar_roof' / f'{frame}.bin'
    lidar_path.parent.mkdir(parents=True)
    np.asarray([[1.0, 2.0, 5.0, 0.5]], dtype=np.float32).tofile(lidar_path)

    calibration = [
        [100.0, 0.0, 96.0, 0.0],
        [0.0, 100.0, 51.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    label = {
        'lane_num': 1,
        'lanes': [[[1.0, 2.0, 5.0], [1.0, 2.0, 10.0]]],
        'calibration': calibration,
    }
    label_path = root / 'ONCE_3DLanes/val' / sequence / 'cam01' / f'{frame}.json'
    _write_json(label_path, label)

    info = {
        'meta_info': {'image_size': [192, 102], 'point_feature_num': 4},
        'calib': {
            'cam01': {
                'cam_to_velo': np.eye(4).tolist(),
                'cam_intrinsic': np.asarray(calibration)[:, :3].tolist(),
                'distortion': [0.0] * 5,
            },
        },
        'frames': [{
            'sequence_id': sequence,
            'frame_id': frame,
            'pose': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }],
    }
    _write_json(root / '3D_infos/data' / sequence / f'{sequence}.json', info)

    split_dir = root / 'LaneLoom_splits'
    split_dir.mkdir(parents=True)
    (split_dir / 'val.txt').write_text(
        f'/{sequence}/cam01/{frame}.jpg\n', encoding='utf-8')
    return root


def _dataset_cfg(root):
    pipeline = [
        dict(type='LoadONCEImages', num_views=1,
             image_augmentation_mode='none', image_augmentation_recipes=[]),
        dict(type='ResizeONCEImages', keep_ratio=True),
        dict(type='NormalizeONCEImages', mean=[0.0, 0.0, 0.0],
             std=[1.0, 1.0, 1.0], to_rgb=True),
        dict(type='LoadONCEPoints', load_dim=4, use_dim=4),
        dict(type='PointsRangeFilter',
             point_cloud_range=[-10.0, 0.0, -5.0, 10.0, 50.0, 5.0]),
        dict(type='LoadONCEAnnotations'),
        dict(type='PackInputs', keys=[
            'img', 'lidar', 'img_metas', 'gt_lanes_3d',
            'gt_lanes_visibility', 'gt_lanes_category', 'raw_label',
        ]),
    ]
    return dict(
        dataset_name='once',
        data_label_dir=str(root / 'ONCE_3DLanes'),
        data_image_dir=str(root / '3D_images/data'),
        lidar_processed_dir=str(root / '3D_lidars/data'),
        info_dir=str(root / '3D_infos/data'),
        split_dir=str(root / 'LaneLoom_splits'),
        train_split='train_p0p1',
        val_split='val',
        test_split='test',
        org_h=102,
        org_w=192,
        resize_h=72,
        resize_w=96,
        anchor_y_steps=[5.0, 10.0],
        train_pipeline=pipeline,
        val_pipeline=pipeline,
        num_category=1,
        max_lanes=8,
        class_names=['lane'],
    )


def test_once_dataset_aligns_camera_lanes_and_lidar_in_ground_axes(tmp_path):
    root = _build_once_fixture(tmp_path)
    dataset = DATASETS.build(dict(
        type='ONCEDataset',
        split='val',
        temporal_sampling_distance=False,
        smooth_3d_gt=False,
        gt_outlier_residual=0.75,
        dataset_cfg=_dataset_cfg(root),
        modality=dict(use_camera=True, use_lidar=True),
        test_mode=True,
    ))

    sample = dataset[0]
    assert sample['img'].shape == (1, 1, 3, 72, 96)
    assert sample['lidar'].shape == (1, 4)
    np.testing.assert_allclose(
        sample['lidar'][0].numpy(),
        [1.0, 5.0, -2.0, 0.5], atol=1e-6)

    visible = sample['gt_lanes_visibility'][0].bool().numpy()
    np.testing.assert_array_equal(visible, [True, True])
    np.testing.assert_allclose(
        sample['gt_lanes_3d'][0].numpy(),
        [[1.0, 5.0, -2.0], [1.0, 10.0, -2.0]], atol=1e-6)
    assert sample['gt_lanes_category'][0].item() == 0
    assert sample['img_metas']['scene_name'] == '000001'
    assert sample['img_metas']['current_key'] == '100'


def test_once_config_preserves_cam01_aspect_ratio_in_network_input(tmp_path):
    root = _build_once_fixture(tmp_path)
    config_path = (
        Path(__file__).resolve().parents[3]
        / 'configs/main/laneloom_once_image_lidar_Temporal.py')
    cfg = Config.fromfile(config_path)
    assert cfg.load_dim == 4
    assert cfg.use_dim == 4
    assert cfg.point_features == ['x', 'y', 'z', 'intensity']
    assert cfg.model.pts_bev_encoder.point_features == cfg.point_features
    dataset_cfg = _dataset_cfg(root)
    dataset_cfg.update(
        org_h=cfg.org_h,
        org_w=cfg.org_w,
        resize_h=cfg.resize_h,
        resize_w=cfg.resize_w,
    )
    dataset = DATASETS.build(dict(
        type='ONCEDataset',
        split='val',
        temporal_sampling_distance=False,
        smooth_3d_gt=False,
        gt_outlier_residual=0.75,
        dataset_cfg=dataset_cfg,
        modality=dict(use_camera=True, use_lidar=True),
        test_mode=True,
    ))

    sample = dataset[0]

    assert (cfg.org_h, cfg.org_w) == (1020, 1920)
    assert (cfg.resize_h, cfg.resize_w) == (510, 960)
    assert sample['img'].shape[-2:] == (510, 960)


def test_once_dataset_rejects_temporal_gt_augmentation(tmp_path):
    root = _build_once_fixture(tmp_path)
    with pytest.raises(ValueError, match='temporal GT'):
        DATASETS.build(dict(
            type='ONCEDataset',
            split='val',
            temporal_sampling_distance=(10.0, 5.0),
            smooth_3d_gt=False,
            gt_outlier_residual=0.75,
            dataset_cfg=_dataset_cfg(root),
            modality=dict(use_camera=True, use_lidar=True),
            test_mode=True,
        ))
