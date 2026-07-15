import os
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from PIL import Image, ImageDraw

from LaneFormer.datasets.pipelines import (
    Albu,
    Compose,
    PointShuffle,
    PointsRangeFilter,
)
from LaneFormer.datasets.pipelines.transforms import A


def _load_cfg():
    config_path = (
        Path(__file__).resolve().parents[3]
        / 'configs'
        / 'laneformer_openlane_image_lidar.py'
    )
    return Config.fromfile(str(config_path))


def _require_openlane_data(cfg):
    required_paths = [
        Path(cfg.dataset_cfg.data_label_dir),
        Path(cfg.dataset_cfg.data_image_dir),
        Path(cfg.dataset_cfg.lidar_processed_dir),
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        pytest.skip('OpenLane/Waymo data is not available: {}'.format(missing_paths))


def _default_openlane_image(cfg, filename):
    return str(
        Path(cfg.dataset_cfg.data_image_dir)
        / 'training'
        / 'segment-8603916601243187272_540_000_560_000_with_camera_labels'
        / filename)


def _build_transform_from_cfg(cfg):
    cfg = dict(cfg)
    transform_type = cfg.pop('type')
    if transform_type == 'PointsRangeFilter':
        return PointsRangeFilter(**cfg)
    if transform_type == 'PointShuffle':
        return PointShuffle(**cfg)
    if transform_type == 'Albu':
        return Albu(**cfg)
    raise ValueError(f'Unsupported pipeline transform in test: {transform_type}')


def _as_rows(points):
    return {tuple(row.tolist()) for row in points}


def test_openlane_pipelines_declare_the_complete_data_flow_in_order():
    cfg = _load_cfg()

    assert [stage['type'] for stage in cfg.train_pipeline] == [
        'LoadTemporalOpenLaneImages',
        'ResizeOpenLaneImages',
        'Albu',
        'NormalizeOpenLaneImages',
        'LoadPoseAlignedOpenLanePoints',
        'PointsRangeFilter',
        'PointShuffle',
        'LoadOpenLaneAnnotations',
        'PackLaneFormerInputs',
    ]
    assert [stage['type'] for stage in cfg.test_pipeline] == [
        'LoadTemporalOpenLaneImages',
        'ResizeOpenLaneImages',
        'NormalizeOpenLaneImages',
        'LoadPoseAlignedOpenLanePoints',
        'PointsRangeFilter',
        'LoadOpenLaneAnnotations',
        'PackLaneFormerInputs',
    ]


def test_points_range_filter_keeps_points_inside_range():
    points = np.array([
        [0.0, 5.0, 0.0, 0.1, 0.2, 0.0],
        [-60.0, 5.0, 0.0, 0.1, 0.2, 0.0],
        [0.0, -1.0, 0.0, 0.1, 0.2, 0.0],
        [50.0, 101.0, 5.0, 0.1, 0.2, 0.0],
    ], dtype=np.float32)

    transform = PointsRangeFilter(point_cloud_range=[-50, 0, -5, 50, 101, 5])
    results = transform(dict(points=points.copy()))

    assert results['points'].shape == (2, 6)
    assert np.allclose(results['points'], points[[0, 3]])


def test_point_shuffle_preserves_points():
    points = np.arange(60, dtype=np.float32).reshape(10, 6)
    np.random.seed(0)
    shuffled = PointShuffle()(dict(points=points.copy()))['points']

    assert shuffled.shape == points.shape
    assert _as_rows(shuffled) == _as_rows(points)
    assert not np.array_equal(shuffled, points)


def test_compose_applies_pipeline_in_order():
    points = np.array([
        [0.0, 5.0, 0.0, 0.1, 0.2, 0.0],
        [1.0, 6.0, 0.0, 0.1, 0.2, 0.0],
        [80.0, 6.0, 0.0, 0.1, 0.2, 0.0],
    ], dtype=np.float32)
    pipeline = Compose([
        PointsRangeFilter(point_cloud_range=[-50, 0, -5, 50, 101, 5]),
        PointShuffle(),
    ])

    np.random.seed(0)
    results = pipeline(dict(points=points.copy()))

    assert results['points'].shape == (2, 6)
    assert _as_rows(results['points']).issubset(_as_rows(points[:2]))


def test_albu_transform_updates_image():
    if A is None:
        pytest.skip('albumentations is not installed.')

    img = np.stack([
        np.tile(np.arange(4, dtype=np.uint8), (3, 1)),
        np.tile(np.arange(10, 14, dtype=np.uint8), (3, 1)),
        np.tile(np.arange(20, 24, dtype=np.uint8), (3, 1)),
    ], axis=-1)
    transform = Albu([
        dict(
            type='OneOf',
            transforms=[dict(type='HorizontalFlip', p=1.0)],
            p=1.0),
    ])

    results = transform(dict(img=img.copy()))

    assert np.array_equal(results['img'], img[:, ::-1, :])


def test_configured_point_pipeline_processes_loaded_openlane_points():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 8
    dataset = DATASETS.build(dataset_cfg)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        points = sample['lidar'].numpy()
        if points.shape[0] == 0:
            continue

        transforms = [
            _build_transform_from_cfg(transform_cfg)
            for transform_cfg in cfg.train_pipeline
            if transform_cfg['type'] in ('PointsRangeFilter', 'PointShuffle')
        ]
        processed = Compose(transforms)(dict(points=points.copy()))['points']
        pc_range = np.asarray(cfg.point_cloud_range, dtype=np.float32)

        assert processed.ndim == 2
        assert processed.shape[1] == points.shape[1]
        assert np.all(processed[:, 0] >= pc_range[0])
        assert np.all(processed[:, 1] >= pc_range[1])
        assert np.all(processed[:, 2] >= pc_range[2])
        assert np.all(processed[:, 0] <= pc_range[3])
        assert np.all(processed[:, 1] <= pc_range[4])
        assert np.all(processed[:, 2] <= pc_range[5])
        break
    else:
        pytest.skip('No non-empty LiDAR points found in sampled OpenLane data.')


def test_configured_multi_aug_views_load_one_shared_gt_sample():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
    dataset_cfg['dataset_cfg'].update(
        num_views=6,
        view_mode='multi_aug_views',
    )
    dataset = DATASETS.build(dataset_cfg)
    sample = dataset[0]

    assert sample['img'].shape[1] == 6
    assert sample['img_metas']['view_names'] == [
        'original',
        'brightness',
        'fog',
        'blur',
        'occlusion',
        'random',
    ]
    assert sample['gt_lanes_3d'].shape[0] == cfg.max_lanes
    assert sample['gt_lanes_uv'].shape[0] == cfg.max_lanes
    assert sample['gt_lanes_visibility'].shape[0] == cfg.max_lanes
    assert not torch.allclose(sample['img'][0, 0], sample['img'][0, 1])


def test_configured_random_aug_views_use_same_random_aug_policy():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
    dataset_cfg['dataset_cfg'].update(
        num_views=6,
        view_mode='random_aug',
    )
    dataset = DATASETS.build(dataset_cfg)
    sample = dataset[0]

    assert sample['img'].shape[1] == 6
    assert torch.all(sample['img_metas']['view_mask'])
    assert any(
        not torch.allclose(sample['img'][0, 0], sample['img'][0, view_idx])
        for view_idx in range(1, 6)
    )


def test_random_aug_view_mode_ignores_surrounding_spaces():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
    dataset_cfg['dataset_cfg'].update(
        num_views=1,
        view_mode='random_aug ',
    )
    dataset = DATASETS.build(dataset_cfg)
    sample = dataset[0]

    assert sample['img_metas']['view_mode'] == 'random_aug'
    assert sample['img'].shape[1] == 1


def test_random_aug_replay_syncs_temporal_queue_augmentation():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
    train_pipeline = [dict(stage) for stage in cfg.train_pipeline]
    for stage in train_pipeline:
        if stage['type'] == 'Albu':
            stage['transforms'] = [
                dict(
                    type='CoarseDropout',
                    max_holes=8,
                    max_height=96,
                    max_width=96,
                    min_holes=8,
                    min_height=96,
                    min_width=96,
                    fill_value=0,
                    p=1.0),
            ]
    dataset_cfg['dataset_cfg'].update(
        num_views=1,
        view_mode='random_aug',
        train_pipeline=train_pipeline)
    dataset = DATASETS.build(dataset_cfg)
    label_info = dataset._read_json(dataset.label_files[0])
    scene_name = dataset.label_files[0].parent.name
    frame_key = dataset.label_files[0].stem[:12]
    image_path = dataset._find_image_path(label_info, scene_name, frame_key)
    random_aug_replays = [None] * dataset.num_views

    first_views, _ = dataset._load_image_views(
        image_path, random_aug_replays=random_aug_replays)
    replayed_views, _ = dataset._load_image_views(
        image_path, random_aug_replays=random_aug_replays)

    assert random_aug_replays[0] is not None
    assert torch.allclose(first_views, replayed_views)


def test_multi_aug_views_seed_syncs_temporal_queue_augmentation():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset_cfg['dataset_cfg'] = dict(dataset_cfg['dataset_cfg'])
    dataset_cfg['dataset_cfg'].update(
        num_views=6,
        view_mode='multi_aug_views',
    )
    dataset = DATASETS.build(dataset_cfg)
    label_info = dataset._read_json(dataset.label_files[0])
    scene_name = dataset.label_files[0].parent.name
    frame_key = dataset.label_files[0].stem[:12]
    image_path = dataset._find_image_path(label_info, scene_name, frame_key)
    view_aug_seeds = [11, 22, 33, 44, 55, 66]

    first_views, _ = dataset._load_image_views(
        image_path, view_aug_seeds=view_aug_seeds)
    replayed_views, _ = dataset._load_image_views(
        image_path, view_aug_seeds=view_aug_seeds)

    assert torch.allclose(first_views, replayed_views)
    assert not torch.allclose(first_views[0], first_views[1])


# ---------------------------------------------------network input visualization start---------------------------------------------------
def _make_tensor_img_canvas(tensor: torch.Tensor,
                            denorm=True,
                            row_labels=None,
                            col_labels=None):
    """Build one PIL canvas from network input image tensor.

    Supported shapes:
        [T, V, 3, H, W]    one dataset sample
        [1, T, V, 3, H, W] one dataloader batch with batch_size=1
    """
    if tensor.ndim == 6:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    assert tensor.ndim == 5
    assert tensor.shape[2] == 3
    if denorm:
        assert torch.min(tensor) < 0

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    numpy_array = tensor.detach().permute(0, 1, 3, 4, 2).cpu().numpy()
    time_steps, view_num, height, width, _ = numpy_array.shape
    if denorm:
        numpy_array = numpy_array * std[None, None, None, None, :] + mean[None, None, None, None, :]
    numpy_array = np.clip(numpy_array, 0, 255).astype('uint8')

    canvas = Image.new('RGB', (width * view_num, height * time_steps), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for time_idx in range(time_steps):
        for view_idx in range(view_num):
            current_image = Image.fromarray(numpy_array[time_idx, view_idx])
            x0 = view_idx * width
            y0 = time_idx * height
            canvas.paste(current_image, (x0, y0))
            if row_labels is not None or col_labels is not None:
                row_name = row_labels[time_idx] if row_labels else f't{time_idx}'
                col_name = col_labels[view_idx] if col_labels else f'view{view_idx}'
                draw.text((x0 + 8, y0 + 8), f'{row_name} {col_name}', fill=(255, 0, 0))
    return canvas


def show_tensor_img(tensor: torch.Tensor,
                    denorm=True,
                    save_path=None,
                    show=False,
                    row_labels=None,
                    col_labels=None):
    canvas = _make_tensor_img_canvas(
        tensor,
        denorm=denorm,
        row_labels=row_labels,
        col_labels=col_labels)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_path)
    if show:
        canvas.show()
    return canvas


def _select_visualization_sample(dataset, image_path=None):
    if image_path:
        image_path = Path(image_path).expanduser().resolve()
        scene_name = image_path.parent.name
        frame_key = image_path.stem[:12]
        for idx, label_path in enumerate(dataset.label_files):
            if label_path.parent.name == scene_name and label_path.stem[:12] == frame_key:
                return dataset[idx], idx
        raise FileNotFoundError(
            f'Cannot find OpenLane label for image sample: {image_path}')

    for idx in range(len(dataset)):
        candidate = dataset[idx]
        frame_keys = candidate['img_metas']['frame_keys']
        if len(set(frame_keys)) == len(frame_keys):
            return candidate, idx
    return dataset[0], 0


def _current_frame_pil(sample):
    """Denormalize the current front-view network input to a PIL image."""
    image = sample['img'][-1, 0].detach().permute(1, 2, 0).cpu().numpy()
    image = image * np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image += np.array([123.675, 116.28, 103.53], dtype=np.float32)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def _draw_ground_lanes_on_current_image(image, lanes, valid_mask, lane2img,
                                        point_color, memory_mask=None):
    """Project current-ground lane anchors through the current calibration."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    lane2img = np.asarray(lane2img, dtype=np.float32)
    lanes = lanes.detach().cpu().numpy()
    valid_mask = valid_mask.detach().cpu().numpy().astype(bool)
    memory_mask = (
        None if memory_mask is None else
        memory_mask.detach().cpu().numpy().astype(bool))
    width, height = image.size
    for lane_idx, lane in enumerate(lanes):
        valid = valid_mask[lane_idx]
        if not np.any(valid):
            continue
        points = np.concatenate(
            [lane, np.ones((lane.shape[0], 1), dtype=np.float32)], axis=1)
        projected = (lane2img @ points.T).T
        depth = projected[:, 2]
        uv = np.zeros((lane.shape[0], 2), dtype=np.float32)
        positive = depth > 1e-5
        uv[positive] = projected[positive, :2] / depth[positive, None]
        keep = valid & positive & (uv[:, 0] >= 0) & (uv[:, 0] < width) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        previous = None
        for point_idx in np.where(keep)[0]:
            xy = tuple(np.round(uv[point_idx]).astype(int))
            is_memory = memory_mask is not None and memory_mask[lane_idx, point_idx]
            color = (255, 80, 30) if is_memory else point_color
            if previous is not None:
                draw.line((previous, xy), fill=color, width=2)
            draw.ellipse(
                (xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3),
                fill=color)
            previous = xy
    return image


def _draw_dataloader_uv_lanes(image, lanes_uv, valid_mask,
                               color=(40, 255, 80)):
    """Draw the 2D lane labels returned by OpenLaneDataset on one frame."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    lanes_uv = lanes_uv.detach().cpu().numpy()
    valid_mask = valid_mask.detach().cpu().numpy().astype(bool)
    width, height = image.size
    for lane_idx, lane_uv in enumerate(lanes_uv):
        keep = valid_mask[lane_idx]
        keep &= (lane_uv[:, 0] >= 0) & (lane_uv[:, 0] < width)
        keep &= (lane_uv[:, 1] >= 0) & (lane_uv[:, 1] < height)
        previous = None
        for point_idx in np.where(keep)[0]:
            xy = tuple(np.round(lane_uv[point_idx]).astype(int))
            if previous is not None:
                draw.line((previous, xy), fill=color, width=2)
            draw.ellipse(
                (xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3), fill=color)
            previous = xy
    return image


def _draw_json_uv_lanes(image, label_info, resize_w, resize_h,
                        org_w=1920, org_h=1280, color=(40, 255, 80)):
    """Draw raw JSON 2D lane annotation after the network resize transform.

    OpenLane JSON may store a different sampling density for ``uv`` and
    ``xyz``.  This is intentionally a raw-annotation visualization rather
    than the 3D-anchor target used by the lane regression head.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    scale = np.array(
        [float(resize_w) / org_w, float(resize_h) / org_h], dtype=np.float32)
    width, height = image.size
    for lane in label_info.get('lane_lines', []):
        uv = np.asarray(lane.get('uv', []), dtype=np.float32)
        if uv.ndim == 2 and uv.shape[0] == 2:
            uv = uv.T
        if uv.ndim != 2 or uv.shape[1] != 2:
            continue
        uv = uv * scale
        keep = np.isfinite(uv).all(axis=1)
        keep &= (uv[:, 0] >= 0) & (uv[:, 0] < width)
        keep &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
        previous = None
        for point_idx in np.where(keep)[0]:
            xy = tuple(np.round(uv[point_idx]).astype(int))
            if previous is not None:
                draw.line((previous, xy), fill=color, width=2)
            draw.ellipse(
                (xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3), fill=color)
            previous = xy
    return image


@pytest.mark.skipif(
    os.getenv('LANEFORMER_VIS_INPUT') != '1',
    reason='Set LANEFORMER_VIS_INPUT=1 to save one OpenLane network-input preview.')
def test_visualize_one_openlane_network_input_sample():
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)

    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    image_path = os.getenv('LANEFORMER_VIS_IMAGE_PATH')
    if image_path is None:
        dataset_cfg['max_samples'] = 128
    dataset = DATASETS.build(dataset_cfg)

    sample, sample_index = _select_visualization_sample(dataset, image_path=image_path)

    output_path = Path(os.getenv(
        'LANEFORMER_VIS_OUTPUT',
        'LaneFormer/tests/test_data/test_pipelines/openlane_network_input_sample.jpg'))
    canvas = show_tensor_img(
        sample['img'],
        denorm=True,
        save_path=output_path,
        show=os.getenv('LANEFORMER_SHOW_INPUT') == '1')

    assert output_path.is_file()
    assert sample_index >= 0
    assert canvas.size[0] > 0 and canvas.size[1] > 0


@pytest.mark.skipif(
    os.getenv('LANEFORMER_VIS_GT') != '1',
    reason='Set LANEFORMER_VIS_GT=1 to save per-frame dataloader GT overlays.')
def test_visualize_openlane_network_input_with_gt():
    """Save the temporal input queue with each frame's own 2D lane GT."""
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    image_path = os.getenv('LANEFORMER_VIS_IMAGE_PATH') or _default_openlane_image(
        cfg, '155250133677391100.jpg')
    dataset = DATASETS.build(dataset_cfg)
    sample, sample_index = _select_visualization_sample(dataset, image_path)
    scene_name = sample['img_metas']['scene_name']
    label_scene = dataset._load_label_scene(scene_name)
    frame_keys = sample['img_metas']['frame_keys']

    panels = []
    for frame_idx, frame_key in enumerate(frame_keys):
        label_info = dataset._read_json(label_scene['frames'][frame_key])
        frame_img = sample['img'][frame_idx, 0].detach().permute(1, 2, 0)
        frame_img = frame_img.cpu().numpy()
        frame_img = frame_img * np.array([58.395, 57.12, 57.375], dtype=np.float32)
        frame_img += np.array([123.675, 116.28, 103.53], dtype=np.float32)
        panel = Image.fromarray(np.clip(frame_img, 0, 255).astype(np.uint8))
        panels.append(_draw_json_uv_lanes(
            panel, label_info, dataset.resize_w, dataset.resize_h,
            dataset.org_w, dataset.org_h))

    width, height = panels[0].size
    canvas = Image.new('RGB', (width, height * len(panels)), (255, 255, 255))
    for frame_idx, panel in enumerate(panels):
        canvas.paste(panel, (0, frame_idx * height))
    output_path = Path(os.getenv(
        'LANEFORMER_VIS_GT_OUTPUT',
        'LaneFormer/tests/test_data/test_pipelines/'
        'openlane_network_input_gt_sample.jpg'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    assert output_path.is_file()
    assert sample_index >= 0


@pytest.mark.skipif(
    os.getenv('LANEFORMER_VIS_3D_GT') != '1',
    reason='Set LANEFORMER_VIS_3D_GT=1 to save per-frame 3D-GT projections.')
def test_visualize_openlane_network_input_with_3d_gt():
    """Save each queue image overlaid with its own 3D anchor GT projection."""
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    image_path = os.getenv('LANEFORMER_VIS_IMAGE_PATH') or _default_openlane_image(
        cfg, '155250133617394700.jpg')
    dataset = DATASETS.build(dataset_cfg)
    sample, sample_index = _select_visualization_sample(dataset, image_path)
    scene_name = sample['img_metas']['scene_name']
    label_scene = dataset._load_label_scene(scene_name)
    frame_keys = sample['img_metas']['frame_keys']

    panels = []
    for frame_idx, frame_key in enumerate(frame_keys):
        label_info = dataset._read_json(label_scene['frames'][frame_key])
        frame_gt = dataset._parse_openlane_gt(label_info)
        frame_img = sample['img'][frame_idx, 0].detach().permute(1, 2, 0)
        frame_img = frame_img.cpu().numpy()
        frame_img = frame_img * np.array([58.395, 57.12, 57.375], dtype=np.float32)
        frame_img += np.array([123.675, 116.28, 103.53], dtype=np.float32)
        panel = Image.fromarray(np.clip(frame_img, 0, 255).astype(np.uint8))
        frame_meta = sample['img_metas']['queue'][frame_idx]
        panels.append(_draw_ground_lanes_on_current_image(
            panel, frame_gt['gt_lanes_3d'], frame_gt['gt_lanes_visibility'],
            frame_meta['lane2img'][0], point_color=(40, 255, 80)))

    width, height = panels[0].size
    canvas = Image.new('RGB', (width, height * len(panels)), (255, 255, 255))
    for frame_idx, panel in enumerate(panels):
        canvas.paste(panel, (0, frame_idx * height))
    output_path = Path(os.getenv(
        'LANEFORMER_VIS_3D_GT_OUTPUT',
        'LaneFormer/tests/test_data/test_pipelines/'
        'openlane_network_input_3d_gt_bezier.jpg'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    assert output_path.is_file()
    assert sample_index >= 0


@pytest.mark.skipif(
    os.getenv('LANEFORMER_VIS_TEMPORAL_GT') != '1',
    reason='Set LANEFORMER_VIS_TEMPORAL_GT=1 to save a temporal-GT comparison.')
def test_visualize_temporal_gt_enhancement_comparison():
    """Save current-GT versus pose-aligned temporal-GT on one current image."""
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['temporal_gt_augmentation'] = True
    image_path = os.getenv('LANEFORMER_TEMPORAL_GT_IMAGE_PATH') or _default_openlane_image(
        cfg, '155250133677391100.jpg')
    dataset = DATASETS.build(dataset_cfg)
    sample, sample_index = _select_visualization_sample(dataset, image_path)
    assert 'gt_lanes_3d_enhanced' in sample

    base_image = _current_frame_pil(sample)
    lane2img = sample['img_metas']['current']['lane2img'][0]
    current_panel = _draw_ground_lanes_on_current_image(
        base_image, sample['gt_lanes_3d'], sample['gt_lanes_visibility'],
        lane2img, point_color=(40, 255, 80))
    enhanced_panel = _draw_ground_lanes_on_current_image(
        base_image, sample['gt_lanes_3d_enhanced'],
        sample['gt_lanes_visibility_enhanced'], lane2img,
        point_color=(40, 190, 255), memory_mask=sample['gt_lanes_memory_mask'])

    width, height = base_image.size
    canvas = Image.new('RGB', (width, height * 2), (255, 255, 255))
    canvas.paste(current_panel, (0, 0))
    canvas.paste(enhanced_panel, (0, height))

    output_path = Path(os.getenv(
        'LANEFORMER_TEMPORAL_GT_OUTPUT',
        'LaneFormer/tests/test_data/test_pipelines/'
        'openlane_temporal_gt_bezier.jpg'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    assert output_path.is_file()
    assert sample_index >= 0
    assert sample['gt_lanes_memory_mask'].sum() >= 0
# ----------------------------------------------------network input visualization end----------------------------------------------------
