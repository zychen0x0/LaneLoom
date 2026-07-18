import json
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


def _raw_visible_ground_lanes(dataset, label_info):
    """Return visible raw JSON lanes in ground coordinates, without y sorting."""
    lane_cam_to_ground = dataset._build_calib(label_info)['lane_cam_to_ground']
    lanes = []
    for lane_info in label_info.get('lane_lines', []):
        lane = np.asarray(lane_info.get('xyz', []), dtype=np.float32)
        if lane.ndim != 2 or lane.shape[0] != 3:
            continue
        visibility = np.asarray(
            lane_info.get('visibility', np.ones(lane.shape[1])),
            dtype=np.float32)
        lane_h = np.vstack((
            lane, np.ones((1, lane.shape[1]), dtype=np.float32)))
        ground = (lane_cam_to_ground @ lane_h)[:3].T.astype(np.float32)
        lanes.append(ground[visibility > 0])
    return lanes


def _fit_ideal_scene_q0(lanes, visibility, pc_range, anchor_y_steps,
                        num_queries=50, max_curvature=0.005,
                        max_grade=0.05):
    """Reproduce the GT-fitted, single-mode LiDAR Q0 upper bound."""
    lanes = np.asarray(lanes, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=np.float32) > 0.5
    lane_curvatures = []
    lane_grades = []
    for lane, valid in zip(lanes, visibility):
        if valid.sum() < 4:
            continue
        points = lane[valid]
        y = points[:, 1]
        x_design = np.stack((np.ones_like(y), y, 0.5 * y ** 2), axis=-1)
        z_design = np.stack((np.ones_like(y), y), axis=-1)
        try:
            lane_curvatures.append(float(
                np.linalg.lstsq(x_design, points[:, 0], rcond=None)[0][2]))
            lane_grades.append(float(
                np.linalg.lstsq(z_design, points[:, 2], rcond=None)[0][1]))
        except np.linalg.LinAlgError:
            continue

    curvature = float(np.clip(
        np.median(lane_curvatures) if lane_curvatures else 0.0,
        -max_curvature, max_curvature))
    grade = float(np.clip(
        np.median(lane_grades) if lane_grades else 0.0,
        -max_grade, max_grade))
    x_min, x_max = float(pc_range[0]), float(pc_range[3])
    cell_width = (x_max - x_min) / num_queries
    start_x = x_min + (np.arange(num_queries, dtype=np.float32) + 0.5) * cell_width
    y = np.asarray(anchor_y_steps, dtype=np.float32)
    x = start_x[:, None] + 0.5 * curvature * y[None, :] ** 2
    z = np.broadcast_to(grade * y[None, :], x.shape)
    q0 = np.stack((x, np.broadcast_to(y[None, :], x.shape), z), axis=-1)
    return q0.astype(np.float32), curvature, grade


def _cubic_fit_error(lane, valid, anchor_y_steps):
    """Measure the best error attainable by the decoder's cubic x(y), z(y)."""
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() < 2:
        return None
    y = np.asarray(anchor_y_steps, dtype=np.float32)[valid]
    span = float(anchor_y_steps[-1] - anchor_y_steps[0])
    t = (y - float(anchor_y_steps[0])) / span
    basis = np.stack((
        (1 - t) ** 3,
        3 * (1 - t) ** 2 * t,
        3 * (1 - t) * t ** 2,
        t ** 3,
    ), axis=-1)
    target = np.asarray(lane, dtype=np.float32)[valid][:, (0, 2)]
    try:
        controls = np.linalg.lstsq(basis, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    residual = np.linalg.norm(basis @ controls - target, axis=-1)
    return float(residual.mean())


def _nearest_q0_error(lane, valid, q0):
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() < 2:
        return None
    target = np.asarray(lane, dtype=np.float32)[valid][:, (0, 2)]
    candidates = q0[:, valid][:, :, (0, 2)]
    mean_distance = np.linalg.norm(
        candidates - target[None, :, :], axis=-1).mean(axis=-1)
    return float(mean_distance.min())


def _y_reversal_count(points):
    if len(points) < 3:
        return 0
    delta_y = np.diff(np.asarray(points)[:, 1])
    direction = np.sign(delta_y[np.abs(delta_y) > 1e-3])
    return int(np.sum(direction[1:] != direction[:-1]))


def _draw_bev_gt_q0(raw_lanes, packed_lanes, packed_visibility, q0,
                    pc_range, title, size=(380, 640)):
    """Draw raw GT, fixed-y GT and the ideal single-mode Q0 in BEV."""
    width, height = size
    margin = 36
    image = Image.new('RGB', size, (18, 18, 22))
    draw = ImageDraw.Draw(image)
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    x_max, y_max = float(pc_range[3]), float(pc_range[4])

    def xy(point):
        x, y = float(point[0]), float(point[1])
        u = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
        v = height - margin - (y - y_min) / (y_max - y_min) * (
            height - 2 * margin)
        return int(round(u)), int(round(v))

    for x in range(int(x_min), int(x_max) + 1, 5):
        color = (65, 65, 72) if x == 0 else (38, 38, 44)
        draw.line((xy((x, y_min)), xy((x, y_max))), fill=color, width=1)
    for y in range(int(y_min), int(y_max) + 1, 10):
        color = (65, 65, 72) if y == 0 else (38, 38, 44)
        draw.line((xy((x_min, y)), xy((x_max, y))), fill=color, width=1)

    for query in q0:
        points = [xy(point) for point in query]
        if len(points) >= 2:
            draw.line(points, fill=(58, 66, 82), width=1)
    for lane in raw_lanes:
        points = [xy(point) for point in lane]
        if len(points) >= 2:
            draw.line(points, fill=(50, 220, 120), width=3)
    for lane, valid in zip(packed_lanes, packed_visibility):
        points = [xy(point) for point in lane[np.asarray(valid) > 0.5]]
        if len(points) >= 2:
            draw.line(points, fill=(255, 145, 45), width=2)

    draw.rectangle((0, 0, width, 31), fill=(8, 8, 10))
    draw.text((8, 5), title, fill=(245, 245, 245))
    draw.rectangle((6, height - 27, width - 6, height - 5), fill=(8, 8, 10))
    draw.text(
        (10, height - 23), 'raw GT  fixed-y GT  ideal shared-mode Q0',
        fill=(220, 220, 220))
    draw.line((10, height - 9, 38, height - 9), fill=(50, 220, 120), width=3)
    draw.line((92, height - 9, 120, height - 9), fill=(255, 145, 45), width=2)
    draw.line((190, height - 9, 218, height - 9), fill=(58, 66, 82), width=1)
    return image


def _gt_audit_image_paths(cfg):
    configured = os.getenv('LANEFORMER_GT_AUDIT_IMAGES')
    if configured:
        return [
            Path(path).expanduser().resolve()
            for path in configured.split(os.pathsep) if path.strip()
        ]
    image_root = Path(cfg.dataset_cfg.data_image_dir)
    samples = [
        ('segment-18418533015310829002_480_000_500_000_with_camera_labels',
         '155008811255146600.jpg'),
        ('segment-17987556068410436875_520_610_540_610_with_camera_labels',
         '151355503476086500.jpg'),
        ('segment-16534202648288984983_900_000_920_000_with_camera_labels',
         '155934386699046500.jpg'),
        ('segment-16534202648288984983_900_000_920_000_with_camera_labels',
         '155934386888974400.jpg'),
        ('segment-16534202648288984983_900_000_920_000_with_camera_labels',
         '155934387138996700.jpg'),
        ('segment-16534202648288984983_900_000_920_000_with_camera_labels',
         '155934387579106400.jpg'),
    ]
    return [image_root / 'training' / scene / filename
            for scene, filename in samples]


def _label_path_for_image(cfg, image_path):
    image_root = Path(cfg.dataset_cfg.data_image_dir).resolve()
    relative = image_path.resolve().relative_to(image_root)
    split, scene_name = relative.parts[:2]
    label_dir = Path(cfg.dataset_cfg.data_label_dir) / split / scene_name
    exact = label_dir / f'{image_path.stem}.json'
    if exact.is_file():
        return exact
    matches = sorted(label_dir.glob(f'{image_path.stem[:12]}*.json'))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Cannot uniquely find label for {image_path}: {matches}')
    return matches[0]


def _audit_one_gt_sample(dataset, cfg, image_path):
    label_path = _label_path_for_image(cfg, image_path)
    label_info = dataset._read_json(label_path)
    raw_lanes = _raw_visible_ground_lanes(dataset, label_info)
    packed = dataset._parse_openlane_gt(label_info)
    packed_lanes = packed['gt_lanes_3d'].numpy()
    packed_visibility = packed['gt_lanes_visibility'].numpy()
    prior_cfg = cfg.model.lane_head.anchor_prior
    q0, curvature, grade = _fit_ideal_scene_q0(
        packed_lanes, packed_visibility, cfg.point_cloud_range,
        dataset.anchor_y_steps, num_queries=prior_cfg.num_anchors,
        max_curvature=prior_cfg.max_curvature,
        max_grade=prior_cfg.max_grade)

    lane_reports = []
    for lane_idx, raw_lane in enumerate(raw_lanes):
        x_span = float(np.ptp(raw_lane[:, 0])) if len(raw_lane) else 0.0
        y_span = float(np.ptp(raw_lane[:, 1])) if len(raw_lane) else 0.0
        represented = (
            lane_idx < len(packed_visibility)
            and int((packed_visibility[lane_idx] > 0.5).sum()) >= 2)
        lane_report = dict(
            lane_index=lane_idx,
            raw_visible_points=int(len(raw_lane)),
            raw_x_span_m=x_span,
            raw_y_span_m=y_span,
            transverse=bool(x_span > y_span and x_span > 1.0),
            y_reversal_count=_y_reversal_count(raw_lane),
            fixed_y_anchor_count=(
                int((packed_visibility[lane_idx] > 0.5).sum())
                if lane_idx < len(packed_visibility) else 0),
            represented=represented,
            ideal_q0_mean_distance_m=None,
            cubic_q2_mean_fit_error_m=None,
        )
        if represented:
            lane_report['ideal_q0_mean_distance_m'] = _nearest_q0_error(
                packed_lanes[lane_idx], packed_visibility[lane_idx], q0)
            lane_report['cubic_q2_mean_fit_error_m'] = _cubic_fit_error(
                packed_lanes[lane_idx], packed_visibility[lane_idx],
                dataset.anchor_y_steps)
        lane_reports.append(lane_report)

    q0_errors = [lane['ideal_q0_mean_distance_m'] for lane in lane_reports
                 if lane['ideal_q0_mean_distance_m'] is not None]
    cubic_errors = [lane['cubic_q2_mean_fit_error_m'] for lane in lane_reports
                    if lane['cubic_q2_mean_fit_error_m'] is not None]
    report = dict(
        image_path=str(image_path),
        label_path=str(label_path),
        raw_lane_count=len(raw_lanes),
        represented_lane_count=sum(lane['represented'] for lane in lane_reports),
        transverse_raw_lane_count=sum(lane['transverse'] for lane in lane_reports),
        unrepresented_raw_lane_count=sum(
            lane['raw_visible_points'] >= 2 and not lane['represented']
            for lane in lane_reports),
        ideal_scene_curvature=curvature,
        ideal_scene_grade=grade,
        ideal_q0_mean_distance_m=(
            float(np.mean(q0_errors)) if q0_errors else None),
        ideal_q0_max_mean_distance_m=(max(q0_errors) if q0_errors else None),
        cubic_q2_mean_fit_error_m=(
            float(np.mean(cubic_errors)) if cubic_errors else None),
        lanes=lane_reports,
    )

    image = Image.open(image_path).convert('RGB').resize(
        (dataset.resize_w, dataset.resize_h), Image.BILINEAR)
    image = _draw_json_uv_lanes(
        image, label_info, dataset.resize_w, dataset.resize_h,
        dataset.org_w, dataset.org_h, color=(50, 220, 120))
    calib = dataset._build_calib(label_info)
    lane2img = dataset._image_resize_matrix() @ calib['ground_to_raw_img']
    image = _draw_ground_lanes_on_current_image(
        image, packed['gt_lanes_3d'], packed['gt_lanes_visibility'],
        lane2img, point_color=(255, 145, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(8, 8, 10))
    draw.text(
        (8, 5),
        f'{image_path.parent.name}/{image_path.name}  '
        f'raw={report["raw_lane_count"]} fixed-y={report["represented_lane_count"]}',
        fill=(245, 245, 245))
    q0_mean = report['ideal_q0_mean_distance_m']
    q0_max = report['ideal_q0_max_mean_distance_m']
    q0_summary = (
        'no represented GT' if q0_mean is None else
        f'Q0 mean/max: {q0_mean:.2f}/{q0_max:.2f} m')
    bev = _draw_bev_gt_q0(
        raw_lanes, packed_lanes, packed_visibility, q0,
        cfg.point_cloud_range, q0_summary,
        size=(380, dataset.resize_h))
    return report, image, bev


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


@pytest.mark.skipif(
    os.getenv('LANEFORMER_AUDIT_GT') != '1',
    reason='Set LANEFORMER_AUDIT_GT=1 to audit raw GT, fixed-y GT and Q0.')
def test_audit_openlane_gt_representation_and_q0_coverage():
    """Save a multi-sample GT/Q0 report for long-tail OpenLane scenes.

    The image panel compares raw JSON UV labels (green) with the fixed-y 3D
    training target (orange).  The BEV panel additionally shows the optimistic
    Q0 obtained by fitting the legacy shared curvature/grade baseline to GT.
    """
    np.random.seed(0)
    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset = DATASETS.build(dataset_cfg)

    image_paths = _gt_audit_image_paths(cfg)
    missing = [str(path) for path in image_paths if not path.is_file()]
    assert not missing, f'GT audit images do not exist: {missing}'

    reports = []
    rows = []
    for image_path in image_paths:
        report, image_panel, bev_panel = _audit_one_gt_sample(
            dataset, cfg, image_path)
        reports.append(report)
        row = Image.new(
            'RGB',
            (image_panel.width + bev_panel.width, image_panel.height),
            (255, 255, 255))
        row.paste(image_panel, (0, 0))
        row.paste(bev_panel, (image_panel.width, 0))
        rows.append(row)

    canvas = Image.new(
        'RGB', (max(row.width for row in rows), sum(row.height for row in rows)),
        (255, 255, 255))
    y_offset = 0
    for row in rows:
        canvas.paste(row, (0, y_offset))
        y_offset += row.height

    output_path = Path(os.getenv(
        'LANEFORMER_GT_AUDIT_OUTPUT',
        'tests/test_data/test_pipelines/show_load_img/'
        'openlane_gt_q0_audit.jpg'))
    report_path = Path(os.getenv(
        'LANEFORMER_GT_AUDIT_REPORT', str(output_path.with_suffix('.json'))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')

    assert output_path.is_file()
    assert report_path.is_file()

    max_q0_distance = os.getenv('LANEFORMER_GT_AUDIT_MAX_Q0_DISTANCE')
    if max_q0_distance is not None:
        threshold = float(max_q0_distance)
        violations = [
            (report['image_path'], lane['lane_index'],
             lane['ideal_q0_mean_distance_m'])
            for report in reports for lane in report['lanes']
            if (lane['ideal_q0_mean_distance_m'] is not None
                and lane['ideal_q0_mean_distance_m'] > threshold)
        ]
        assert not violations, (
            f'Ideal shared-mode Q0 exceeds {threshold} m: {violations}')


@pytest.mark.skipif(
    os.getenv('LANEFORMER_VIS_PATHOLOGICAL_GT') != '1',
    reason='Set LANEFORMER_VIS_PATHOLOGICAL_GT=1 to visualize a pathological fit.')
def test_visualize_pathological_openlane_bezier_refit():
    """Visualize a rejected hard-inlier refit and the guarded GT output."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset = DATASETS.build(dataset_cfg)

    scene = 'segment-4058410353286511411_3980_000_4000_000_with_camera_labels'
    image_path = Path(os.getenv(
        'LANEFORMER_PATHOLOGICAL_GT_IMAGE',
        str(Path(cfg.dataset_cfg.data_image_dir) / 'training' / scene
            / '155077875506292500.jpg')))
    assert image_path.is_file(), image_path
    label_path = _label_path_for_image(cfg, image_path)
    label_info = dataset._read_json(label_path)
    lane_index = int(os.getenv('LANEFORMER_PATHOLOGICAL_GT_LANE_INDEX', '0'))
    records = dataset._parse_lane_records(label_info)
    assert 0 <= lane_index < len(records)

    raw = np.asarray(records[lane_index]['ground_raw'], dtype=np.float64)
    raw = raw[np.argsort(raw[:, 1])]
    _, unique_indices = np.unique(raw[:, 1], return_index=True)
    raw = raw[unique_indices]
    y_min, y_max = raw[0, 1], raw[-1, 1]

    def basis(y_values):
        t = (np.asarray(y_values, dtype=np.float64) - y_min) / (y_max - y_min)
        return np.stack((
            (1 - t) ** 3,
            3 * (1 - t) ** 2 * t,
            3 * (1 - t) * t ** 2,
            t ** 3,
        ), axis=-1)

    design = basis(raw[:, 1])
    target = raw[:, (0, 2)]
    controls_all = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = np.linalg.norm(design @ controls_all - target, axis=-1)
    inliers = residual <= dataset.gt_outlier_residual
    assert inliers.sum() >= 4
    controls_refit = np.linalg.lstsq(
        design[inliers], target[inliers], rcond=None)[0]

    start_idx, end_idx = dataset._anchor_index_bounds(y_min, y_max)
    anchor_y = dataset.anchor_y_steps[start_idx:end_idx + 1]
    dense_y = np.linspace(anchor_y[0], anchor_y[-1], 400)

    def curve(controls, y_values):
        xz = basis(y_values) @ controls
        return np.column_stack((xz[:, 0], y_values, xz[:, 1]))

    all_curve = curve(controls_all, dense_y)
    refit_curve = curve(controls_refit, dense_y)
    current_smoothed = np.asarray(records[lane_index]['ground'], dtype=np.float64)
    cond_all = float(np.linalg.cond(design))
    cond_refit = float(np.linalg.cond(design[inliers]))

    calib = dataset._build_calib(label_info)
    ground_to_image = np.asarray(calib['ground_to_raw_img'], dtype=np.float64)

    def project(points):
        homogeneous = np.column_stack((points, np.ones(len(points))))
        projected = (ground_to_image @ homogeneous.T).T
        keep = projected[:, 2] > 1e-5
        uv = np.full((len(points), 2), np.nan, dtype=np.float64)
        uv[keep] = projected[keep, :2] / projected[keep, 2:3]
        return uv

    figure = plt.figure(figsize=(20, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    image_axis = figure.add_subplot(grid[0, 0])
    residual_axis = figure.add_subplot(grid[0, 1])
    info_axis = figure.add_subplot(grid[0, 2])
    local_axis = figure.add_subplot(grid[1, 0])
    global_axis = figure.add_subplot(grid[1, 1])
    height_axis = figure.add_subplot(grid[1, 2])

    image_axis.imshow(Image.open(image_path).convert('RGB'))
    for points, color, label, width in (
            (raw, '#32d583', 'raw GT points', 0),
            (all_curve, '#22d3ee', 'first fit: all points', 2),
            (refit_curve, '#ef4444', 'rejected second fit', 2),
            (current_smoothed, '#a855f7', 'guarded training GT', 2)):
        uv = project(points)
        if width:
            image_axis.plot(uv[:, 0], uv[:, 1], color=color, linewidth=width,
                            label=label)
        else:
            image_axis.scatter(uv[:, 0], uv[:, 1], color=color, s=30,
                               label=label)
    image_axis.set_xlim(0, dataset.org_w)
    image_axis.set_ylim(dataset.org_h, 0)
    image_axis.set_title('Camera projection (pathological curve is mostly off-image)')
    image_axis.legend(loc='lower left', fontsize=8)
    image_axis.axis('off')

    residual_axis.scatter(
        raw[inliers, 1], residual[inliers], color='#22c55e', s=50,
        label='kept as inlier')
    residual_axis.scatter(
        raw[~inliers, 1], residual[~inliers], color='#f59e0b', s=50,
        label='removed')
    residual_axis.axhline(
        dataset.gt_outlier_residual, color='#ef4444', linestyle='--',
        label=f'threshold={dataset.gt_outlier_residual:.2f}m')
    residual_axis.set_xlabel('ground y (m)')
    residual_axis.set_ylabel('first-fit x/z residual (m)')
    residual_axis.set_title('Hard residual filtering removes y-range support')
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(fontsize=8)

    info_axis.axis('off')
    info_axis.text(
        0, 1,
        '\n'.join((
            'PATHOLOGICAL OPENLANE GT BEZIER REFIT',
            f'image: {image_path.name}',
            f'lane index: {lane_index}',
            '',
            f'raw y support: [{y_min:.2f}, {y_max:.2f}] m',
            f'raw points: {len(raw)}',
            f'hard inliers: {int(inliers.sum())}',
            f'inlier y support: [{raw[inliers, 1].min():.2f}, '
            f'{raw[inliers, 1].max():.2f}] m',
            '',
            f'condition, all points: {cond_all:.1f}',
            f'condition, hard inliers: {cond_refit:.1f}',
            '',
            f'raw |x|max: {np.abs(raw[:, 0]).max():.2f} m',
            f'raw |z|max: {np.abs(raw[:, 2]).max():.2f} m',
            f'refit |x|max: {np.abs(refit_curve[:, 0]).max():.2f} m',
            f'refit |z|max: {np.abs(refit_curve[:, 2]).max():.2f} m',
            '',
            'green: raw GT',
            'cyan: first all-point cubic fit',
            'red: rejected hard-inlier cubic refit',
            'purple: guarded training GT',
        )),
        va='top', family='monospace', fontsize=11)

    def draw_bev(axis, x_limits, title):
        axis.plot(all_curve[:, 0], all_curve[:, 1], color='#22d3ee',
                  linewidth=2, label='first fit')
        axis.plot(refit_curve[:, 0], refit_curve[:, 1], color='#ef4444',
                  linewidth=2, label='rejected second refit')
        axis.plot(current_smoothed[:, 0], current_smoothed[:, 1],
                  color='#a855f7', linewidth=2, linestyle='--',
                  label='guarded GT')
        axis.scatter(raw[inliers, 0], raw[inliers, 1], color='#22c55e',
                     s=34, label='raw inlier')
        axis.scatter(raw[~inliers, 0], raw[~inliers, 1], color='#f59e0b',
                     marker='x', s=55, label='raw removed')
        axis.set_xlim(*x_limits)
        axis.set_ylim(anchor_y[0] - 0.5, anchor_y[-1] + 0.5)
        axis.set_xlabel('ground x (m)')
        axis.set_ylabel('forward y (m)')
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    draw_bev(local_axis, (-15, 5), 'Local BEV: raw GT and stable first fit')
    global_x_min = min(-30.0, float(refit_curve[:, 0].min()) * 1.02)
    global_x_max = max(30.0, float(refit_curve[:, 0].max()) * 1.02)
    draw_bev(global_axis, (global_x_min, global_x_max),
             'Expanded BEV: second refit explodes')

    height_axis.plot(dense_y, all_curve[:, 2], color='#22d3ee', linewidth=2,
                     label='first fit')
    height_axis.plot(dense_y, refit_curve[:, 2], color='#ef4444', linewidth=2,
                     label='second refit')
    height_axis.scatter(raw[inliers, 1], raw[inliers, 2], color='#22c55e',
                        s=34, label='raw inlier')
    height_axis.scatter(raw[~inliers, 1], raw[~inliers, 2], color='#f59e0b',
                        marker='x', s=55, label='raw removed')
    height_axis.set_xlabel('forward y (m)')
    height_axis.set_ylabel('ground z (m)')
    height_axis.set_title('Height profile: pathological vertical oscillation')
    height_axis.grid(alpha=0.25)
    height_axis.legend(fontsize=8)

    figure.suptitle(
        'Guard rejects the second fit when inliers collapse to the lane end',
        fontsize=16)
    output_path = Path(os.getenv(
        'LANEFORMER_PATHOLOGICAL_GT_OUTPUT',
        'tests/test_data/test_pipelines/show_load_img/'
        'openlane_pathological_bezier_refit.jpg'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

    assert output_path.is_file()
    assert cond_refit > 1e4
    assert np.abs(current_smoothed[:, 0]).max() < 20


@pytest.mark.skipif(
    os.getenv('LANEFORMER_VIS_BENEFICIAL_REFIT') != '1',
    reason='Set LANEFORMER_VIS_BENEFICIAL_REFIT=1 to compare a useful refit.')
def test_visualize_beneficial_openlane_bezier_refit():
    """Visualize a real lane where stable inlier refitting reduces residual."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = _load_cfg()
    _require_openlane_data(cfg)
    dataset_cfg = dict(cfg.train_dataloader['dataset'])
    dataset_cfg['max_samples'] = 1
    dataset = DATASETS.build(dataset_cfg)

    scene = 'segment-16238753252899859750_1340_000_1360_000_with_camera_labels'
    image_path = Path(cfg.dataset_cfg.data_image_dir) / 'training' / scene \
        / '154906697149727700.jpg'
    label_info = dataset._read_json(_label_path_for_image(cfg, image_path))
    lane_index = 1
    record = dataset._parse_lane_records(label_info)[lane_index]
    raw = np.asarray(record['ground_raw'], dtype=np.float64)
    raw = raw[np.argsort(raw[:, 1])]
    _, unique_indices = np.unique(raw[:, 1], return_index=True)
    raw = raw[unique_indices]
    y_min, y_max = raw[0, 1], raw[-1, 1]

    def basis(y_values):
        t = (np.asarray(y_values, dtype=np.float64) - y_min) / (y_max - y_min)
        return np.stack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                         3 * (1 - t) * t ** 2, t ** 3), axis=-1)

    design = basis(raw[:, 1])
    target = raw[:, (0, 2)]
    first_controls = np.linalg.lstsq(design, target, rcond=None)[0]
    first_residual = np.linalg.norm(design @ first_controls - target, axis=-1)
    inliers = first_residual <= dataset.gt_outlier_residual
    assert dataset._inliers_support_stable(raw[:, 1], design, inliers)
    second_controls = np.linalg.lstsq(
        design[inliers], target[inliers], rcond=None)[0]
    second_residual = np.linalg.norm(
        design[inliers] @ second_controls - target[inliers], axis=-1)
    first_rmse = float(np.sqrt(np.mean(first_residual[inliers] ** 2)))
    second_rmse = float(np.sqrt(np.mean(second_residual ** 2)))

    dense_y = np.linspace(y_min, y_max, 500)
    first_xz = basis(dense_y) @ first_controls
    second_xz = basis(dense_y) @ second_controls
    first_curve = np.column_stack((first_xz[:, 0], dense_y, first_xz[:, 1]))
    second_curve = np.column_stack((second_xz[:, 0], dense_y, second_xz[:, 1]))

    ground_to_image = np.asarray(
        dataset._build_calib(label_info)['ground_to_raw_img'], dtype=np.float64)

    def project(points):
        projected = (ground_to_image @ np.column_stack(
            (points, np.ones(len(points)))).T).T
        uv = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid = projected[:, 2] > 1e-5
        uv[valid] = projected[valid, :2] / projected[valid, 2:3]
        return uv

    figure, axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    image_axis, bev_axis, residual_axis, height_axis = axes.ravel()
    image_axis.imshow(Image.open(image_path).convert('RGB'))
    for curve, color, name in (
            (first_curve, '#22d3ee', 'first fit'),
            (second_curve, '#a855f7', 'accepted second fit')):
        uv = project(curve)
        image_axis.plot(uv[:, 0], uv[:, 1], color=color, linewidth=2, label=name)
    raw_uv = project(raw)
    image_axis.scatter(raw_uv[inliers, 0], raw_uv[inliers, 1], color='#22c55e',
                       s=22, label='inlier GT')
    image_axis.scatter(raw_uv[~inliers, 0], raw_uv[~inliers, 1], color='#f59e0b',
                       marker='x', s=38, label='removed outlier')
    image_axis.set_xlim(0, dataset.org_w)
    image_axis.set_ylim(dataset.org_h, 0)
    image_axis.axis('off')
    image_axis.set_title('Camera projection')
    image_axis.legend(fontsize=8)

    bev_axis.plot(first_curve[:, 0], first_curve[:, 1], color='#22d3ee',
                  linewidth=2, label='first fit')
    bev_axis.plot(second_curve[:, 0], second_curve[:, 1], color='#a855f7',
                  linewidth=2, label='accepted second fit')
    bev_axis.scatter(raw[inliers, 0], raw[inliers, 1], color='#22c55e', s=20)
    bev_axis.scatter(raw[~inliers, 0], raw[~inliers, 1], color='#f59e0b',
                     marker='x', s=38)
    bev_axis.set_xlabel('ground x (m)')
    bev_axis.set_ylabel('forward y (m)')
    bev_axis.set_title('BEV geometry')
    bev_axis.grid(alpha=0.25)
    bev_axis.legend(fontsize=8)

    residual_axis.scatter(raw[inliers, 1], first_residual[inliers],
                          color='#22d3ee', s=25, label='first fit residual')
    residual_axis.scatter(raw[inliers, 1], second_residual,
                          color='#a855f7', s=25, label='second fit residual')
    residual_axis.scatter(raw[~inliers, 1], first_residual[~inliers],
                          color='#f59e0b', marker='x', s=38,
                          label='rejected raw points')
    residual_axis.axhline(dataset.gt_outlier_residual, color='#ef4444',
                          linestyle='--', label='0.75m threshold')
    residual_axis.set_xlabel('forward y (m)')
    residual_axis.set_ylabel('x/z residual (m)')
    residual_axis.set_title(
        f'Inlier RMSE: {first_rmse:.3f}m -> {second_rmse:.3f}m')
    residual_axis.grid(alpha=0.25)
    residual_axis.legend(fontsize=8)

    height_axis.plot(dense_y, first_curve[:, 2], color='#22d3ee',
                     linewidth=2, label='first fit')
    height_axis.plot(dense_y, second_curve[:, 2], color='#a855f7',
                     linewidth=2, label='accepted second fit')
    height_axis.scatter(raw[inliers, 1], raw[inliers, 2], color='#22c55e', s=20)
    height_axis.scatter(raw[~inliers, 1], raw[~inliers, 2], color='#f59e0b',
                        marker='x', s=38)
    height_axis.set_xlabel('forward y (m)')
    height_axis.set_ylabel('ground z (m)')
    height_axis.set_title('Vertical profile')
    height_axis.grid(alpha=0.25)
    height_axis.legend(fontsize=8)

    figure.suptitle(
        'Beneficial robust refit: reject isolated points while preserving support',
        fontsize=15)
    output_path = Path(os.getenv(
        'LANEFORMER_BENEFICIAL_REFIT_OUTPUT',
        'tests/test_data/test_pipelines/show_load_img/'
        'openlane_beneficial_bezier_refit.jpg'))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

    assert output_path.is_file()
    assert second_rmse < first_rmse
    assert np.allclose(
        record['ground'][:, (0, 2)],
        basis(record['ground'][:, 1]) @ second_controls,
        atol=1e-4)
# ----------------------------------------------------network input visualization end----------------------------------------------------
