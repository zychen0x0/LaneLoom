import bisect
import glob
import json
import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from mmengine.registry import FUNCTIONS
from mmdet3d.registry import DATASETS

from .pipelines import GridMask

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - kept for minimal environments
    cKDTree = None

try:
    import albumentations as A
except ImportError:
    A = None


# ---------------------------------------------------path utils start---------------------------------------------------
def _as_path(path):
    return Path(path).expanduser().resolve()


def _timestamp_key(path_or_name):
    return Path(path_or_name).stem[:12]


def _to_homogeneous(points_xyz):
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1).T


WAYMO_VEHICLE_TO_GROUND_AXES = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
], dtype=np.float32)

NORM_CAMERA_TO_GROUND_AXES = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

NORM_CAMERA_TO_GROUND_ROT = NORM_CAMERA_TO_GROUND_AXES[:3, :3]


def _vehicle_transform_to_ground(transform):
    """Express a vehicle-axis rigid transform in OpenLane ground axes."""
    ground_from_vehicle = np.eye(4, dtype=np.float32)
    ground_from_vehicle[:3, :3] = WAYMO_VEHICLE_TO_GROUND_AXES
    return (ground_from_vehicle
            @ np.asarray(transform, dtype=np.float32).reshape(4, 4)
            @ np.linalg.inv(ground_from_vehicle)).astype(np.float32)


def _build_cam_to_ground_lane(cam_to_vehicle):
    cam_to_vehicle = np.asarray(cam_to_vehicle, dtype=np.float32).reshape(4, 4)
    cam_to_ground_lane = cam_to_vehicle.copy()
    cam_to_ground_lane[:3, :3] = (
        WAYMO_VEHICLE_TO_GROUND_AXES @ cam_to_ground_lane[:3, :3])
    cam_to_ground_lane[0:2, 3] = 0.0
    return cam_to_ground_lane.astype(np.float32)


def _build_norm_cam_to_vehicle(cam_to_vehicle):
    cam_to_vehicle = np.asarray(cam_to_vehicle, dtype=np.float32).reshape(4, 4)
    norm_cam_to_vehicle = cam_to_vehicle.copy()
    norm_cam_to_vehicle[:3, :3] = (
        cam_to_vehicle[:3, :3] @ NORM_CAMERA_TO_GROUND_ROT)
    return norm_cam_to_vehicle.astype(np.float32)


def _build_cam_to_ground(cam_to_vehicle):
    return (
        _build_cam_to_ground_lane(cam_to_vehicle)
        @ NORM_CAMERA_TO_GROUND_AXES
    ).astype(np.float32)


def _pad_3x4_to_4x4(matrix):
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = np.asarray(matrix, dtype=np.float32).reshape(3, 4)
    return out


class WaterDropBlur:
    """Synthetic water drops/smudges on camera lens for rainy-scene robustness."""

    def __init__(self,
                 num_drops=6,
                 min_radius=28,
                 max_radius=90,
                 blur_radius=10,
                 alpha=0.42,
                 tint_color=None,
                 tint_alpha=0.0,
                 p=1.0):
        self.num_drops = int(num_drops)
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)
        self.blur_radius = float(blur_radius)
        self.alpha = float(alpha)
        self.tint_color = tint_color
        self.tint_alpha = float(tint_alpha)
        self.p = float(p)

    def __call__(self, image, **kwargs):
        if np.random.rand() > self.p:
            return dict(image=image)

        pil_img = Image.fromarray(image).convert('RGB')
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        width, height = pil_img.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for _ in range(self.num_drops):
            radius_x = np.random.randint(self.min_radius, self.max_radius + 1)
            radius_y = np.random.randint(self.min_radius, self.max_radius + 1)
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            draw.ellipse(
                (center_x - radius_x, center_y - radius_y,
                 center_x + radius_x, center_y + radius_y),
                fill=int(255 * self.alpha))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=self.blur_radius * 0.8))
        if self.tint_color is not None and self.tint_alpha > 0:
            tint = Image.new('RGB', (width, height), tuple(self.tint_color))
            blurred = Image.blend(blurred, tint, self.tint_alpha)
        mixed = Image.composite(blurred, pil_img, mask)
        return dict(image=np.asarray(mixed, dtype=np.uint8))


class ColorTint:
    """Directional scene color tint, e.g. warm street lights or red brake lights."""

    def __init__(self, color=(255, 180, 80), alpha=0.25, p=1.0):
        self.color = tuple(color)
        self.alpha = float(alpha)
        self.p = float(p)

    def __call__(self, image, **kwargs):
        if np.random.rand() > self.p:
            return dict(image=image)
        pil_img = Image.fromarray(image).convert('RGB')
        tint = Image.new('RGB', pil_img.size, self.color)
        mixed = Image.blend(pil_img, tint, self.alpha)
        return dict(image=np.asarray(mixed, dtype=np.uint8))
# ----------------------------------------------------path utils end----------------------------------------------------


@DATASETS.register_module()
class OpenLaneDataset(Dataset):
    """OpenLane image + Waymo LiDAR queue loader for LaneLoom.

    Returned tensor shapes:
        img:   [T, 1, 3, H, W], T = history frames + current frame, num_cams = 1
        lidar: [N, 6],          xyz/intensity/elongation/queue_index

    The queue is chronological: oldest history frame first, current frame last.
    LiDAR keeps variable-length points; voxelization regularizes it later.
    """

    METAINFO = dict(dataset_name='OpenLane')
    PIPELINE_STAGE_ORDER = (
        'LoadTemporalOpenLaneImages',
        'ResizeOpenLaneImages',
        'Albu',
        'NormalizeOpenLaneImages',
        'GridMask',
        'LoadPoseAlignedOpenLanePoints',
        'PointsRangeFilter',
        'PointShuffle',
        'LoadOpenLaneAnnotations',
        'PackInputs',
    )
    GT_REFIT_MIN_INLIERS = 6
    GT_REFIT_MIN_SPAN_RATIO = 0.6
    GT_REFIT_ENDPOINT_MARGIN_RATIO = 0.2
    GT_REFIT_MAX_CONDITION = 1e3
    GT_REFIT_MIN_RMSE_IMPROVEMENT = 1e-4

    # ---------------------------------------------------init start---------------------------------------------------
    def __init__(self,
                 dataset_cfg,
                 frames,
                 history_num_frames,
                 history_distance_m,
                 history_distance_step_m,
                 history_fallback_frame_offsets,
                 temporal_gt_augmentation,
                 smooth_3d_gt,
                 gt_outlier_residual,
                 modality,
                 test_mode=False,
                 split=None,
                 max_samples=None):
        self.test_mode = test_mode
        self.modality = modality
        self.dataset_cfg = dataset_cfg
        self.split = self._resolve_split(split or ('val' if test_mode else 'train'))

        self.is_train = self.split == dataset_cfg['train_split']
        self.pipeline_cfg = (
            dataset_cfg['train_pipeline']
            if self.is_train else dataset_cfg['val_pipeline']
        )
        if not isinstance(self.pipeline_cfg, (list, tuple)):
            raise TypeError('OpenLane pipeline must be an ordered list of stages.')
        self.pipeline_cfg = [dict(stage) for stage in self.pipeline_cfg]
        self._validate_pipeline()
        image_load_cfg = self._pipeline_stage('LoadTemporalOpenLaneImages')
        image_resize_cfg = self._pipeline_stage('ResizeOpenLaneImages')
        image_normalize_cfg = self._pipeline_stage('NormalizeOpenLaneImages')
        grid_mask_cfg = next((
            stage for stage in self.pipeline_cfg
            if stage.get('type') == 'GridMask'
        ), None)
        points_load_cfg = self._pipeline_stage('LoadPoseAlignedOpenLanePoints')
        self._pipeline_stage('LoadOpenLaneAnnotations')
        pack_cfg = self._pipeline_stage('PackInputs')
        self.label_root = _as_path(dataset_cfg['data_label_dir'])
        self.image_root = _as_path(dataset_cfg['data_image_dir'])
        self.lidar_root = _as_path(dataset_cfg['lidar_processed_dir'])
        self.load_interval = max(1, int(dataset_cfg.get('load_interval', 1)))

        self.org_h = int(dataset_cfg.get('org_h', 1280))
        self.org_w = int(dataset_cfg.get('org_w', 1920))
        self.img_scale = dataset_cfg.get(
            'img_scale', image_resize_cfg.get('img_scale'))
        if not image_resize_cfg.get('keep_ratio', True):
            raise ValueError('ResizeOpenLaneImages currently requires keep_ratio=True.')
        self.resize_w, self.resize_h = self._resolve_image_size(dataset_cfg)
        self.num_views = int(dataset_cfg.get(
            'num_views', image_load_cfg.get('num_views', 1)))
        self.view_mode = str(dataset_cfg.get(
            'view_mode', image_load_cfg.get('view_mode', 'original'))).strip()
        self.view_aug_cfgs = list(dataset_cfg.get(
            'multi_aug_view_transforms',
            image_load_cfg.get('multi_aug_view_transforms', [])))
        self.mean = np.asarray(
            dataset_cfg.get('mean', image_normalize_cfg['mean']),
            dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(
            dataset_cfg.get('std', image_normalize_cfg['std']),
            dtype=np.float32).reshape(3, 1, 1)

        self.frames = tuple(frames)
        self.history_num_frames = int(history_num_frames)
        self.history_frame_step = 1
        self.history_distance_m = float(history_distance_m)
        self.history_distance_step_m = float(history_distance_step_m)
        self.history_fallback_frame_offsets = tuple(
            int(offset) for offset in history_fallback_frame_offsets)
        if len(self.history_fallback_frame_offsets) != self.history_num_frames:
            raise ValueError(
                'history_fallback_frame_offsets must contain exactly '
                'history_num_frames offsets.')
        if any(offset <= 0 for offset in self.history_fallback_frame_offsets):
            raise ValueError('history fallback offsets must be positive.')
        self.temporal_gt_augmentation = bool(temporal_gt_augmentation)
        self.smooth_3d_gt = bool(smooth_3d_gt)
        self.gt_outlier_residual = float(gt_outlier_residual)
        self.load_dim = int(dataset_cfg.get(
            'load_dim', points_load_cfg.get('load_dim', 6)))
        self.use_dim = dataset_cfg.get(
            'use_dim', points_load_cfg.get('use_dim', self.load_dim))
        self.num_lidar_feat = self._resolve_lidar_feat_dim(self.use_dim)
        self.max_lanes = int(dataset_cfg.get('max_lanes', 20))
        self.num_category = int(dataset_cfg.get('num_category', 21))
        self.metainfo = dict(
            self.METAINFO,
            classes=tuple(dataset_cfg.get('class_names', ())),
            num_category=self.num_category)
        self.anchor_y_steps = np.asarray(
            dataset_cfg.get('anchor_y_steps', np.arange(0, 101, 2)),
            dtype=np.float32)
        self.img_aug = self._build_img_aug()
        self.img_aug_replay = self._build_img_aug(replay=True)
        self.view_aug_names, self.view_aug_transforms = self._build_view_aug()
        self.grid_mask = self._build_grid_mask(grid_mask_cfg)
        self.pts_aug_cfg = [
            stage for stage in self.pipeline_cfg
            if stage['type'] in ('PointsRangeFilter', 'PointShuffle')
        ]
        self.point_cloud_range = self._resolve_point_cloud_range()
        self.shuffle_points = any(
            aug.get('type') == 'PointShuffle' for aug in self.pts_aug_cfg)
        self.pack_keys = tuple(pack_cfg.get('keys', ()))

        self.label_files = self._load_label_files(max_samples=max_samples)
        self.lidar_segment_dirs = self._load_lidar_segment_dirs()
        self._lidar_scene_cache = {}
        self._label_scene_cache = {}
        # Group samplers expect a flag per sample.
        self.flag = np.zeros(len(self.label_files), dtype=np.uint8)
    # ----------------------------------------------------init end----------------------------------------------------

    # ---------------------------------------------------pipeline build start---------------------------------------------------
    def _validate_pipeline(self):
        stage_types = [stage.get('type') for stage in self.pipeline_cfg]
        unknown = [
            stage_type for stage_type in stage_types
            if stage_type not in self.PIPELINE_STAGE_ORDER
        ]
        if unknown:
            raise ValueError(f'Unsupported OpenLane pipeline stages: {unknown}')
        if len(stage_types) != len(set(stage_types)):
            raise ValueError('OpenLane pipeline stages must have unique types.')
        stage_order = {
            stage_type: index
            for index, stage_type in enumerate(self.PIPELINE_STAGE_ORDER)
        }
        configured_order = [stage_order[stage_type] for stage_type in stage_types]
        if configured_order != sorted(configured_order):
            raise ValueError('OpenLane pipeline stages are out of execution order.')

    def _pipeline_stage(self, stage_type):
        matches = [
            stage for stage in self.pipeline_cfg
            if stage.get('type') == stage_type
        ]
        if len(matches) != 1:
            raise ValueError(
                f'OpenLane pipeline requires exactly one {stage_type} stage; '
                f'found {len(matches)}.')
        return matches[0]

    @staticmethod
    def _resolve_lidar_feat_dim(use_dim):
        if isinstance(use_dim, int):
            return int(use_dim)
        return len(use_dim)

    def _resolve_image_size(self, dataset_cfg):
        """Resolve img_scale=(short, long) with keep_ratio."""
        if self.img_scale is None:
            if (
                dataset_cfg.get('resize_w') is None
                or dataset_cfg.get('resize_h') is None
            ):
                raise ValueError('OpenLaneDataset requires img_scale or resize_w/resize_h.')
            return (
                int(dataset_cfg['resize_w']),
                int(dataset_cfg['resize_h']),
            )

        scale = tuple(self.img_scale)
        if len(scale) != 2:
            raise ValueError(f'img_scale should be a 2-tuple, got {self.img_scale}')
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(
            float(max_long_edge) / float(max(self.org_w, self.org_h)),
            float(max_short_edge) / float(min(self.org_w, self.org_h)),
        )
        resize_w = int(self.org_w * scale_factor + 0.5)
        resize_h = int(self.org_h * scale_factor + 0.5)
        return resize_w, resize_h

    def _build_img_aug_transform(self, cfg):
        if A is None:
            raise ImportError('albumentations is required for img_aug type=Albu.')

        cfg = dict(cfg)
        transform_type = cfg.pop('type')
        nested = cfg.pop('transforms', None)
        cfg = {key: value for key, value in cfg.items() if value is not None}

        if transform_type == 'OneOf':
            transforms = [self._build_img_aug_transform(item) for item in nested]
            return A.OneOf(transforms, **cfg)
        if transform_type == 'WaterDropBlur':
            return WaterDropBlur(**cfg)
        if transform_type == 'ColorTint':
            return ColorTint(**cfg)
        if not hasattr(A, transform_type):
            raise ValueError(f'Unsupported albumentations transform: {transform_type}')
        return getattr(A, transform_type)(**cfg)

    def _build_img_aug(self, replay=False):
        albu_cfg = next((
            stage for stage in self.pipeline_cfg
            if stage.get('type') == 'Albu'
        ), None)
        if albu_cfg is None:
            return None
        return self._build_albu_compose(albu_cfg.get('transforms', []), replay=replay)

    def _build_albu_compose(self, transforms, replay=False):
        if not transforms:
            return None
        transforms = [
            self._build_img_aug_transform(item)
            for item in transforms
        ]
        if replay:
            return A.ReplayCompose(transforms)
        return A.Compose(transforms)

    def _build_view_aug(self):
        names = []
        transforms = []
        for view_cfg in self.view_aug_cfgs:
            names.append(view_cfg.get('name', f'view{len(names)}'))
            transforms.append(self._build_albu_compose(view_cfg.get('transforms', [])))
        return names, transforms

    @staticmethod
    def _build_grid_mask(cfg):
        if cfg is None:
            return None
        cfg = dict(cfg)
        cfg.pop('type')
        return GridMask(**cfg).train()

    def _resolve_point_cloud_range(self):
        for aug in self.pts_aug_cfg:
            if aug.get('type') == 'PointsRangeFilter':
                return np.asarray(aug['point_cloud_range'], dtype=np.float32)
        return None

    # ----------------------------------------------------pipeline build end----------------------------------------------------

    # ---------------------------------------------------index build start---------------------------------------------------
    def _resolve_split(self, split):
        if split in ('train', 'training'):
            return self.dataset_cfg['train_split']
        if split in ('val', 'valid', 'validation'):
            return self.dataset_cfg['val_split']
        if split in ('test', 'testing'):
            return self.dataset_cfg['test_split']
        return split

    def _scenario_name(self, label_path):
        """Return an official test scenario, or an empty label for other splits."""
        relative = Path(label_path).relative_to(self.label_root / self.split)
        # validation/training: segment/frame.json; test: scenario/segment/frame.json
        return relative.parts[0] if len(relative.parts) >= 3 else ''

    def _load_label_files(self, max_samples=None):
        split_dir = self.label_root / self.split
        # training/validation use ``split/segment/frame.json`` while the
        # official test split adds a scenario directory, e.g.
        # ``test/merge_split_case/segment/frame.json``.
        label_files = sorted(split_dir.rglob('*.json'))
        if not label_files:
            raise FileNotFoundError(f'No OpenLane json files found under {split_dir}')
        if max_samples is not None:
            label_files = label_files[:int(max_samples)]
        label_files = label_files[::self.load_interval]
        return label_files

    def _load_lidar_segment_dirs(self):
        segment_dirs = sorted(self.lidar_root.glob('segment-*'))
        return {segment_dir.name: segment_dir for segment_dir in segment_dirs}

    def _load_lidar_scene(self, scene_name):
        if scene_name in self._lidar_scene_cache:
            return self._lidar_scene_cache[scene_name]

        segment_dir = self.lidar_segment_dirs.get(scene_name)
        if segment_dir is None:
            self._lidar_scene_cache[scene_name] = None
            return None

        segment_pkl = segment_dir / f'{scene_name}.pkl'
        if not segment_pkl.is_file():
            self._lidar_scene_cache[scene_name] = None
            return None

        with open(segment_pkl, 'rb') as f:
            frame_infos = pickle.load(f)

        frames = {}
        for frame_info in frame_infos:
            metadata = frame_info.get('metadata', {})
            point_cloud = frame_info.get('point_cloud', {})
            timestamp = metadata.get('timestamp_micros')
            sample_idx = point_cloud.get('sample_idx')
            pose = frame_info.get('pose')
            if timestamp is None or sample_idx is None:
                continue
            token = f'{int(sample_idx):04d}.npy'
            lidar_file = segment_dir / token
            if not lidar_file.is_file():
                continue
            frames[str(timestamp)[:12]] = dict(
                token=token,
                lidar_file=lidar_file,
                pose=np.asarray(pose, dtype=np.float32).reshape(4, 4),
                timestamp_micros=int(timestamp),
                sample_idx=int(sample_idx),
            )

        scene = dict(keys=sorted(frames.keys()), frames=frames)
        self._lidar_scene_cache[scene_name] = scene
        return scene

    def _load_label_scene(self, scene_name, scene_dir=None):
        scene_dir = Path(scene_dir or (self.label_root / self.split / scene_name))
        cache_key = str(scene_dir)
        if cache_key in self._label_scene_cache:
            return self._label_scene_cache[cache_key]

        frame_map = {_timestamp_key(path): path for path in sorted(scene_dir.glob('*.json'))}
        scene = dict(keys=sorted(frame_map.keys()), frames=frame_map)
        self._label_scene_cache[cache_key] = scene
        return scene
    # ----------------------------------------------------index build end----------------------------------------------------

    # ------------------------------------------------temporal sampling start-------------------------------------------------
    def _distance_sampling_enabled(self):
        return self.history_distance_m > 0 and self.history_distance_step_m > 0

    def _pose_distance(self, current_pose, history_pose):
        # Vehicle poses are vehicle-to-global. This gives history->current motion in meters.
        rel_pose = np.linalg.inv(history_pose) @ current_pose
        return float(np.linalg.norm(rel_pose[:2, 3]))

    def _sample_history_by_distance(self, scene, current_key):
        keys = scene['keys']
        current_pos = bisect.bisect_left(keys, current_key)
        if current_pos <= 0 or current_key not in scene['frames']:
            return self._trim_history_to_config(scene, current_key, [])

        current_pose = scene['frames'][current_key]['pose']
        candidates = []
        for key in reversed(keys[:current_pos]):
            dist = self._pose_distance(current_pose, scene['frames'][key]['pose'])
            candidates.append((key, dist))

        selected = []
        used = set()
        target = self.history_distance_step_m
        while target <= self.history_distance_m + 1e-6:
            for key, dist in candidates:
                if key not in used and dist >= target:
                    selected.append(key)
                    used.add(key)
                    break
            target += self.history_distance_step_m

        # Do not fill here: the caller must distinguish a successful distance
        # queue from one that needs the explicit t-10/t-5 fallback policy.
        return sorted(selected, key=keys.index)

    def _sample_history_by_frame_offsets(self, scene, current_key):
        """Return t-10/t-5-style fallback keys in chronological order."""
        keys = scene['keys']
        current_pos = bisect.bisect_left(keys, current_key)
        if current_pos <= 0:
            return self._trim_history_to_config(scene, current_key, [])

        selected = []
        for offset in self.history_fallback_frame_offsets:
            # At sequence start the oldest available history is repeated only
            # as a real frame key, so image, pose, calibration and LiDAR stay
            # mutually consistent.
            selected.append(keys[max(0, current_pos - offset)])
        return self._trim_history_to_config(scene, current_key, selected)

    def _sample_history_by_neighbor(self, scene, current_key):
        keys = scene['keys']
        current_pos = bisect.bisect_left(keys, current_key)
        selected = []
        for offset in range(1, self.history_num_frames + 1):
            target_pos = current_pos - offset * self.history_frame_step
            if target_pos < 0:
                break
            selected.append(keys[target_pos])
        return self._trim_history_to_config(scene, current_key, selected)

    def _trim_history_to_config(self, scene, current_key, selected):
        keys = scene['keys']
        current_pos = bisect.bisect_left(keys, current_key)

        selected = [key for key in selected if key in scene['frames']]
        selected = sorted(set(selected), key=keys.index)

        if len(selected) < self.history_num_frames:
            for key in reversed(keys[:current_pos]):
                if key in selected:
                    continue
                selected.insert(0, key)
                if len(selected) >= self.history_num_frames:
                    break
            selected = sorted(set(selected), key=keys.index)

        if len(selected) > self.history_num_frames:
            selected = selected[-self.history_num_frames:]
        while len(selected) < self.history_num_frames:
            # 序列开头没有足够历史时保持固定 T, 用最近可用帧补齐队列左侧。
            selected.insert(0, selected[0] if selected else current_key)
        return selected

    def _sample_frame_queue_keys(self, scene_name, current_key, label_scene=None):
        lidar_scene = self._load_lidar_scene(scene_name)
        label_scene = label_scene or self._load_label_scene(scene_name)
        if lidar_scene is not None:
            # A queue key is valid only when both modalities have that exact
            # frame. This prevents silently pairing a historical LiDAR sweep
            # with the current image/pose/label metadata.
            common_keys = [
                key for key in lidar_scene['keys']
                if key in lidar_scene['frames'] and key in label_scene['frames']
            ]
            if current_key not in common_keys:
                raise KeyError(
                    f'Current frame {scene_name}/{current_key} is not present '
                    'in both LiDAR and OpenLane labels.')
            scene = dict(
                keys=common_keys,
                frames={key: lidar_scene['frames'][key] for key in common_keys})
        else:
            scene = label_scene

        if (
            self._distance_sampling_enabled()
            and lidar_scene is not None
            and current_key in scene['frames']
        ):
            history_keys = self._sample_history_by_distance(scene, current_key)
            if len(history_keys) < self.history_num_frames:
                history_keys = self._sample_history_by_frame_offsets(scene, current_key)
                sampling_mode = 'distance_fallback_offsets'
            else:
                sampling_mode = 'distance'
        else:
            history_keys = self._sample_history_by_neighbor(label_scene, current_key)
            sampling_mode = 'neighbor'

        queue_keys = history_keys + [current_key]
        return queue_keys, sampling_mode, lidar_scene, label_scene
    # -------------------------------------------------temporal sampling end--------------------------------------------------

    # ---------------------------------------------------image load start---------------------------------------------------
    def _read_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.loads(f.readline())

    def _build_calib(self, label_info):
        cam_to_vehicle = np.asarray(
            label_info['extrinsic'], dtype=np.float32).reshape(4, 4)
        camera_intrinsics = np.asarray(
            label_info['intrinsic'], dtype=np.float32).reshape(3, 3)

        lane_cam_to_ground = _build_cam_to_ground_lane(cam_to_vehicle)
        norm_cam_to_ground = _build_cam_to_ground(cam_to_vehicle)
        norm_cam_to_vehicle = _build_norm_cam_to_vehicle(cam_to_vehicle)
        vehicle_to_norm_cam = np.linalg.inv(norm_cam_to_vehicle).astype(np.float32)
        ground_to_norm_cam = np.linalg.inv(norm_cam_to_ground).astype(np.float32)

        vehicle_to_raw_img = (
            camera_intrinsics @ vehicle_to_norm_cam[:3]
        ).astype(np.float32)
        ground_to_raw_img = (
            camera_intrinsics @ ground_to_norm_cam[:3]
        ).astype(np.float32)

        return dict(
            K=camera_intrinsics,
            cam_to_vehicle=cam_to_vehicle,
            vehicle_to_cam=vehicle_to_norm_cam,
            lane_cam_to_ground=lane_cam_to_ground,
            norm_cam_to_ground=norm_cam_to_ground,
            norm_cam_to_vehicle=norm_cam_to_vehicle,
            vehicle_to_norm_cam=vehicle_to_norm_cam,
            vehicle_to_ground=(
                norm_cam_to_ground @ vehicle_to_norm_cam
            ).astype(np.float32),
            ground_to_norm_cam=ground_to_norm_cam,
            vehicle_to_raw_img=vehicle_to_raw_img,
            ground_to_raw_img=ground_to_raw_img,
        )

    def _image_resize_matrix(self):
        sx = float(self.resize_w) / float(self.org_w)
        sy = float(self.resize_h) / float(self.org_h)
        return np.array([
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def _build_frame_meta(self, scene_name, frame_key, label_info,
                          image_path, lidar_scene, current_pose):
        calib = self._build_calib(label_info)
        img_aug_matrix = self._image_resize_matrix()
        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:3, :3] = calib['K']
        cam2img_net = cam2img.copy()
        cam2img_net[:3, :3] = img_aug_matrix @ calib['K']

        vehicle_to_img = (img_aug_matrix @ calib['vehicle_to_raw_img']).astype(np.float32)
        ground_to_img = (img_aug_matrix @ calib['ground_to_raw_img']).astype(np.float32)

        frame_info = None if lidar_scene is None else lidar_scene['frames'].get(frame_key)
        pose = (
            np.asarray(frame_info['pose'], dtype=np.float32).reshape(4, 4)
            if frame_info is not None else
            np.asarray(label_info.get('pose', np.eye(4)), dtype=np.float32).reshape(4, 4)
        )
        if current_pose is None or np.allclose(pose, current_pose, atol=1e-4):
            lidaradj2lidarcurr = np.eye(4, dtype=np.float32)
        else:
            lidaradj2lidarcurr = _vehicle_transform_to_ground(
                np.linalg.inv(current_pose) @ pose)

        pts_filename = (
            str(frame_info['lidar_file'])
            if frame_info is not None and 'lidar_file' in frame_info else None
        )
        sensor2lidar_rotation = calib['cam_to_vehicle'][:3, :3].astype(np.float32)
        sensor2lidar_translation = calib['cam_to_vehicle'][:3, 3].astype(np.float32)
        caminfo = dict(
            sensor2lidar_translation=sensor2lidar_translation,
            sensor2lidar_rotation=sensor2lidar_rotation,
            cam_intrinsic=calib['K'],
        )

        image_paths = [str(image_path)] * self.num_views
        lidar2img = [_pad_3x4_to_4x4(vehicle_to_img) for _ in range(self.num_views)]
        cam2img_list = [cam2img_net.copy() for _ in range(self.num_views)]
        lidar2cam = [calib['vehicle_to_cam'].copy() for _ in range(self.num_views)]
        caminfo_list = [dict(
            sensor2lidar_translation=sensor2lidar_translation.copy(),
            sensor2lidar_rotation=sensor2lidar_rotation.copy(),
            cam_intrinsic=calib['K'].copy(),
        ) for _ in range(self.num_views)]
        lane2img = [_pad_3x4_to_4x4(ground_to_img) for _ in range(self.num_views)]

        return dict(
            # Standard MMDet3D-style sample fields.
            sample_idx=frame_key,
            pts_filename=pts_filename,
            sweeps=[],
            timestamp=(
                float(frame_info['timestamp_micros']) / 1e6
                if frame_info is not None else float(frame_key)
            ),
            timestamp_micros=(
                int(frame_info['timestamp_micros'])
                if frame_info is not None else int(frame_key)
            ),
            ego2global_translation=pose[:3, 3].astype(np.float32),
            ego2global_rotation=pose[:3, :3].astype(np.float32),
            lidar2ego_translation=np.zeros(3, dtype=np.float32),
            lidar2ego_rotation=np.eye(3, dtype=np.float32),
            prev=None,
            next=None,
            scene_token=scene_name,
            frame_idx=frame_key,
            img_filename=image_paths,
            lidar2img=lidar2img,
            cam2img=cam2img_list,
            lidar2cam=lidar2cam,
            caminfo=caminfo_list,

            # Image metadata consumed by image pipelines/backbones.
            scene_name=scene_name,
            ori_shape=(self.org_h, self.org_w, 3),
            img_shape=(self.resize_h, self.resize_w, 3),
            pad_shape=(self.resize_h, self.resize_w, 3),
            img_norm_cfg=dict(mean=self.mean.reshape(-1), std=self.std.reshape(-1), to_rgb=True),
            img_aug_matrix=img_aug_matrix,

            # Temporal alignment field used by queue fusion.
            lidaradj2lidarcurr=lidaradj2lidarcurr,

            # OpenLane-specific lane-coordinate metadata.
            lane2img=lane2img,
            openlane_calib=calib,
        )

    def prepare_input_dict(self, info):
        """Build an OpenLane input dict from one frame metadata record."""
        input_dict = dict(
            sample_idx=info['sample_idx'],
            pts_filename=info['pts_filename'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            prev=info['prev'],
            next=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
        )

        if self.modality.get('use_camera', True):
            input_dict.update(
                dict(
                    img_filename=info['img_filename'],
                    lidar2img=info['lidar2img'],
                    cam2img=info['cam2img'],
                    lidar2cam=info['lidar2cam'],
                    caminfo=info['caminfo'],
                    img_scale=self.img_scale,
                    img_shape=info['img_shape'],
                    pad_shape=info['pad_shape'],
                    img_norm_cfg=info['img_norm_cfg'],
                ))

        return input_dict

    def get_data_info(self, index):
        """Return current-frame metadata with image and LiDAR calibration."""
        current_json_path = self.label_files[index]
        scene_name = current_json_path.parent.name
        current_key = _timestamp_key(current_json_path)
        label_info = self._read_json(current_json_path)
        image_path = self._find_image_path(label_info, scene_name, current_key)
        lidar_scene = self._load_lidar_scene(scene_name)
        current_pose = np.asarray(
            label_info.get('pose', np.eye(4)),
            dtype=np.float32).reshape(4, 4)
        if lidar_scene is not None and current_key in lidar_scene['frames']:
            current_pose = lidar_scene['frames'][current_key]['pose']

        frame_meta = self._build_frame_meta(
            scene_name, current_key, label_info, image_path, lidar_scene, current_pose)
        input_dict = self.prepare_input_dict(frame_meta)
        if not self.test_mode:
            input_dict['ann_info'] = self._parse_openlane_gt(label_info)
        return input_dict

    def _find_image_path(self, label_info, scene_name, frame_key):
        file_path = label_info.get('file_path')
        if file_path:
            image_path = self.image_root / file_path
            if image_path.is_file():
                return image_path

        matches = []
        scene_image_dir = self.image_root / self.split / scene_name
        for suffix in ('jpg', 'jpeg', 'png'):
            matches.extend(glob.glob(str(scene_image_dir / f'{frame_key}*.{suffix}')))
        if not matches:
            raise FileNotFoundError(f'No image found for {scene_name}/{frame_key}')
        return Path(sorted(matches)[0])

    def _load_image_tensor(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h), Image.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
        if self.img_aug is not None:
            array = self.img_aug(image=array)['image']
        array = array.astype(np.float32).transpose(2, 0, 1)
        array = (array - self.mean) / self.std
        return torch.from_numpy(array)

    def _apply_transform_with_seed(self, transform, image, seed):
        if seed is None:
            return transform(image=image)['image']

        numpy_state = np.random.get_state()
        random_state = random.getstate()
        np.random.seed(int(seed))
        random.seed(int(seed))
        try:
            return transform(image=image)['image']
        finally:
            np.random.set_state(numpy_state)
            random.setstate(random_state)

    def _load_image_views(self, image_path, random_aug_replays=None, view_aug_seeds=None):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h), Image.BILINEAR)
        base_array = np.asarray(image, dtype=np.uint8)

        view_tensors = []
        for view_idx in range(self.num_views):
            array = base_array.copy()
            if self.view_mode == 'random_aug' and self.img_aug is not None:
                if random_aug_replays is None or self.img_aug_replay is None:
                    array = self.img_aug(image=array)['image']
                elif random_aug_replays[view_idx] is None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            message='.*could work incorrectly in ReplayMode.*',
                            category=UserWarning)
                        replayed = self.img_aug_replay(image=array)
                    array = replayed['image']
                    random_aug_replays[view_idx] = replayed['replay']
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            message='.*could work incorrectly in ReplayMode.*',
                            category=UserWarning)
                        array = A.ReplayCompose.replay(
                            random_aug_replays[view_idx], image=array)['image']
            elif self.view_mode == 'multi_aug_views':
                if view_idx < len(self.view_aug_transforms):
                    transform = self.view_aug_transforms[view_idx]
                    if transform is not None:
                        seed = (
                            view_aug_seeds[view_idx]
                            if view_aug_seeds is not None and view_idx < len(view_aug_seeds)
                            else None
                        )
                        array = self._apply_transform_with_seed(transform, array, seed)
                elif self.img_aug is not None:
                    seed = (
                        view_aug_seeds[view_idx]
                        if view_aug_seeds is not None and view_idx < len(view_aug_seeds)
                        else None
                    )
                    array = self._apply_transform_with_seed(self.img_aug, array, seed)
            array = array.astype(np.float32).transpose(2, 0, 1)
            array = (array - self.mean) / self.std
            view_tensors.append(torch.from_numpy(array))

        views = torch.stack(view_tensors, dim=0)
        active_mask = torch.ones(self.num_views, dtype=torch.bool)
        return views, active_mask
    # ----------------------------------------------------image load end----------------------------------------------------

    # ---------------------------------------------------gt build start---------------------------------------------------
    def _as_uv(self, uv):
        if uv is None:
            return None
        uv = np.asarray(uv, dtype=np.float32)
        if uv.ndim == 2 and uv.shape[0] == 2:
            uv = uv.T
        if uv.ndim != 2 or uv.shape[1] != 2:
            return None
        return uv

    def _project_lane_cam_to_uv(self, xyz_cam, K):
        proj = (K @ xyz_cam.T).T
        depth = proj[:, 2]
        uv = np.full((xyz_cam.shape[0], 2), np.nan, dtype=np.float32)
        valid = depth > 1e-5
        uv[valid] = proj[valid, :2] / depth[valid, None]
        return uv

    def _smooth_lane_ground(self, points):
        """Fit full-support GT, then sample only anchors inside model support.

        Every visible raw point participates in the first cubic fit, including
        points beyond the model's last y anchor.  A hard-inlier second fit is
        accepted only when its design matrix still covers the original lane
        support and is numerically stable.  Sampling never extrapolates beyond
        either the raw GT support or the model anchor range.

        Returns the sampled points, the retained raw-point mask, and one of
        ``first_fit``, ``robust_refit``, ``linear_fallback``, or ``rejected``.
        """
        points = np.asarray(points, dtype=np.float32)
        if not self.smooth_3d_gt:
            return (
                points,
                np.ones(points.shape[0], dtype=bool),
                'linear_fallback',
            )

        order = np.argsort(points[:, 1])
        sorted_points = points[order]
        _, unique_indices = np.unique(sorted_points[:, 1], return_index=True)
        sorted_points = sorted_points[unique_indices]
        retained_indices = order[unique_indices]
        keep_all = np.zeros(points.shape[0], dtype=bool)
        keep_all[retained_indices] = True
        if sorted_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), keep_all, 'rejected'

        y_min = float(sorted_points[0, 1])
        y_max = float(sorted_points[-1, 1])
        sample_mask = self._anchors_inside_support(y_min, y_max)
        sample_y = self.anchor_y_steps[sample_mask].astype(np.float64)
        if sample_y.size == 0:
            # Preserve the raw annotation in ``ground_raw``/UV audit fields,
            # but do not manufacture a target at the nearest boundary anchor.
            return np.zeros((0, 3), dtype=np.float32), keep_all, 'rejected'

        raw_y = sorted_points[:, 1].astype(np.float64)
        raw_xz = sorted_points[:, (0, 2)].astype(np.float64)

        def linear_fallback():
            sample_x = np.interp(sample_y, raw_y, raw_xz[:, 0])
            sample_z = np.interp(sample_y, raw_y, raw_xz[:, 1])
            return np.column_stack((sample_x, sample_y, sample_z)).astype(np.float32)

        if sorted_points.shape[0] < 4 or y_max - y_min < 1e-4:
            return linear_fallback(), keep_all, 'linear_fallback'

        def bezier_basis(y_values):
            t = (np.asarray(y_values, dtype=np.float64) - y_min) / (y_max - y_min)
            return np.stack(
                [(1.0 - t) ** 3, 3.0 * (1.0 - t) ** 2 * t,
                 3.0 * (1.0 - t) * t ** 2, t ** 3], axis=1)

        design = bezier_basis(raw_y)

        def fit(mask):
            selected_design = design[mask]
            controls, _, rank, _ = np.linalg.lstsq(
                selected_design, raw_xz[mask], rcond=None)
            condition = float(np.linalg.cond(selected_design))
            return controls, int(rank), condition

        all_points = np.ones(sorted_points.shape[0], dtype=bool)
        selected_points = all_points
        fit_status = 'first_fit'
        try:
            controls, first_rank, first_condition = fit(all_points)
            if (first_rank < 4 or not np.isfinite(first_condition)
                    or first_condition > 1e6):
                return linear_fallback(), keep_all, 'linear_fallback'

            predicted = design @ controls
            residual = np.linalg.norm(raw_xz - predicted, axis=1)
            inliers = residual <= self.gt_outlier_residual
            if (not inliers.all()
                    and self._inliers_support_stable(raw_y, design, inliers)):
                refit_controls, refit_rank, refit_condition = fit(inliers)
                first_rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
                refit_residual = np.linalg.norm(
                    raw_xz[inliers] - design[inliers] @ refit_controls, axis=1)
                refit_rmse = float(np.sqrt(np.mean(refit_residual ** 2)))
                rmse_improvement = first_rmse - refit_rmse
                if (refit_rank == 4 and np.isfinite(refit_condition)
                        and refit_condition <= self.GT_REFIT_MAX_CONDITION
                        and rmse_improvement
                        >= self.GT_REFIT_MIN_RMSE_IMPROVEMENT):
                    controls = refit_controls
                    selected_points = inliers
                    fit_status = 'robust_refit'
        except (np.linalg.LinAlgError, ValueError):
            return linear_fallback(), keep_all, 'linear_fallback'

        sample_xz = bezier_basis(sample_y) @ controls
        if not self._bezier_fit_is_physical(sample_xz, sample_y, raw_xz, raw_y):
            return linear_fallback(), keep_all, 'linear_fallback'

        keep = np.zeros(points.shape[0], dtype=bool)
        keep[retained_indices[selected_points]] = True
        result = np.column_stack([sample_xz[:, 0], sample_y, sample_xz[:, 1]])
        return result.astype(np.float32), keep, fit_status

    def _inliers_support_stable(self, raw_y, design, inliers):
        """Whether hard inliers can identify a full-support cubic safely."""
        if int(inliers.sum()) < self.GT_REFIT_MIN_INLIERS:
            return False
        full_span = float(raw_y[-1] - raw_y[0])
        inlier_y = raw_y[inliers]
        if full_span <= 1e-6:
            return False
        if (float(inlier_y[-1] - inlier_y[0]) / full_span
                < self.GT_REFIT_MIN_SPAN_RATIO):
            return False
        endpoint_margin = self.GT_REFIT_ENDPOINT_MARGIN_RATIO * full_span
        if (inlier_y[0] > raw_y[0] + endpoint_margin
                or inlier_y[-1] < raw_y[-1] - endpoint_margin):
            return False
        inlier_design = design[inliers]
        condition = float(np.linalg.cond(inlier_design))
        return bool(
            np.linalg.matrix_rank(inlier_design) == 4
            and np.isfinite(condition)
            and condition <= self.GT_REFIT_MAX_CONDITION)

    @staticmethod
    def _bezier_fit_is_physical(sample_xz, sample_y, raw_xz, raw_y):
        """Reject non-finite or explosive cubic interpolation results."""
        if not np.isfinite(sample_xz).all():
            return False
        linear_xz = np.column_stack((
            np.interp(sample_y, raw_y, raw_xz[:, 0]),
            np.interp(sample_y, raw_y, raw_xz[:, 1])))
        raw_extent = float(np.linalg.norm(np.ptp(raw_xz, axis=0)))
        max_allowed_deviation = max(5.0, 2.0 * raw_extent)
        deviation = np.linalg.norm(sample_xz - linear_xz, axis=1)
        if float(deviation.max(initial=0.0)) > max_allowed_deviation:
            return False

        raw_min = raw_xz.min(axis=0)
        raw_max = raw_xz.max(axis=0)
        margin = np.maximum(np.asarray([5.0, 2.0]), 2.0 * (raw_max - raw_min))
        return bool(
            np.all(sample_xz >= raw_min - margin)
            and np.all(sample_xz <= raw_max + margin))

    def _parse_lane_records(self, label_info, frame_meta=None,
                            is_history=False):
        """Parse visible lane geometry and preserve the JSON lane track id.

        ``global_from_ground`` is intentionally derived through the exact
        camera-to-ground transform used for ordinary training targets.  This
        avoids assuming that OpenLane's ground axes equal Waymo vehicle axes.
        """
        calib = self._build_calib(label_info)
        lane_cam_to_ground = calib['lane_cam_to_ground']
        K = calib['K']
        image_aug = self._image_resize_matrix()
        global_from_ground = None
        if frame_meta is not None:
            global_from_vehicle = np.eye(4, dtype=np.float32)
            global_from_vehicle[:3, :3] = np.asarray(
                frame_meta['ego2global_rotation'], dtype=np.float32)
            global_from_vehicle[:3, 3] = np.asarray(
                frame_meta['ego2global_translation'], dtype=np.float32)
            vehicle_from_ground = (
                calib['cam_to_vehicle'] @ np.linalg.inv(lane_cam_to_ground))
            global_from_ground = (
                global_from_vehicle @ vehicle_from_ground).astype(np.float32)

        records = []
        for lane_info in label_info.get('lane_lines', []):
            lane = np.asarray(lane_info.get('xyz', []), dtype=np.float32)
            if lane.ndim != 2 or lane.shape[0] != 3:
                continue
            visibility = np.asarray(
                lane_info.get('visibility', np.ones(lane.shape[1])),
                dtype=np.float32)

            uv = self._as_uv(lane_info.get('uv'))
            if uv is None or uv.shape[0] != lane.shape[1]:
                uv = self._project_lane_cam_to_uv(lane.T.astype(np.float32), K)

            lane_h = np.vstack([lane, np.ones((1, lane.shape[1]), dtype=np.float32)])
            lane_ground = (lane_cam_to_ground @ lane_h)[:3].T.astype(np.float32)
            visible = visibility > 0
            lane_ground = lane_ground[visible]
            uv = uv[visible]
            if lane_ground.shape[0] > 0:
                order = np.argsort(lane_ground[:, 1])
                lane_ground = lane_ground[order]
                uv = uv[order]
            lane_ground_raw = lane_ground.copy()
            lane_ground, smooth_keep, fit_status = self._smooth_lane_ground(
                lane_ground)
            uv = uv[smooth_keep]
            uv_ground_y = lane_ground_raw[smooth_keep, 1:2]

            category = int(lane_info.get('category', 1))
            if category == 21:
                category = 20
            category = min(max(category, 0), self.num_category - 1)

            uv_h = np.concatenate(
                [uv, np.ones((uv.shape[0], 1), dtype=np.float32)], axis=1)
            uv_net = (uv_h @ image_aug.T)[:, :2].astype(np.float32)
            uv_y = np.concatenate(
                [uv_net, uv_ground_y.astype(np.float32)], axis=1
            ) if uv_net.shape[0] else np.zeros((0, 3), dtype=np.float32)

            track_id = lane_info.get('track_id')
            try:
                track_id = int(track_id) if track_id is not None else -1
            except (TypeError, ValueError):
                track_id = -1
            record = dict(
                ground=lane_ground,
                ground_raw=lane_ground_raw,
                uv=uv_net,
                uvy=uv_y,
                category=category,
                track_id=track_id,
                fit_status=fit_status,
                is_history=bool(is_history))
            if global_from_ground is not None and lane_ground.shape[0] > 0:
                lane_h = np.concatenate(
                    [lane_ground, np.ones((lane_ground.shape[0], 1), dtype=np.float32)],
                    axis=1)
                record['global'] = (
                    global_from_ground @ lane_h.T).T[:, :3].astype(np.float32)
            records.append(record)
        return records

    def _parse_openlane_gt(self, label_info):
        records = self._parse_lane_records(label_info)
        packed = self._pack_lane_targets(
            [record['ground'] for record in records],
            [record['uv'] for record in records],
            [record['uvy'] for record in records],
            [record['category'] for record in records],
            track_ids=[record['track_id'] for record in records])
        # Memory masks belong only to the disabled temporal-GT experiment;
        # the ordinary training target is strictly current-frame GT.
        packed.pop('gt_lanes_memory_mask')
        return packed

    def _pack_lane_targets(self, lanes_3d, lanes_uv, lanes_uvy, categories,
                           track_ids=None, memory_point_masks=None):
        num_y = len(self.anchor_y_steps)
        ground_lanes = np.zeros(
            (self.max_lanes, 3 * num_y + self.num_category), dtype=np.float32)
        gt_lanes_3d = np.zeros((self.max_lanes, num_y, 3), dtype=np.float32)
        gt_lanes_uv = np.zeros((self.max_lanes, num_y, 2), dtype=np.float32)
        gt_lanes_visibility = np.zeros((self.max_lanes, num_y), dtype=np.float32)
        gt_lanes_category = np.zeros((self.max_lanes,), dtype=np.int64)
        gt_lane_track_ids = np.full((self.max_lanes,), -1, dtype=np.int64)
        gt_lanes_memory_mask = np.zeros((self.max_lanes, num_y), dtype=np.float32)

        raw_lanes = []
        packed_lane_idx = 0
        for source_lane_idx, lane in enumerate(lanes_3d):
            if lane.shape[0] < 2:
                continue
            source_y = lane[:, 1].astype(np.float32)
            if self._anchors_inside_support(
                    source_y.min(), source_y.max()).sum() < 2:
                continue
            if packed_lane_idx >= self.max_lanes:
                break

            lane_idx = packed_lane_idx
            packed_lane_idx += 1
            category = int(categories[source_lane_idx])
            gt_lanes_category[lane_idx] = category
            if track_ids is not None:
                gt_lane_track_ids[lane_idx] = int(track_ids[source_lane_idx])
            raw_lanes.append(lane.astype(np.float32))

            y = lane[:, 1].astype(np.float32)
            unique_y, unique_idx = np.unique(y, return_index=True)
            if unique_y.shape[0] < 2:
                continue
            lane = lane[unique_idx]
            valid = self._anchors_inside_support(unique_y.min(), unique_y.max())
            xs = np.interp(self.anchor_y_steps, unique_y, lane[:, 0]).astype(np.float32)
            zs = np.interp(self.anchor_y_steps, unique_y, lane[:, 2]).astype(np.float32)

            gt_lanes_3d[lane_idx, :, 0] = xs
            gt_lanes_3d[lane_idx, :, 1] = self.anchor_y_steps
            gt_lanes_3d[lane_idx, :, 2] = zs
            gt_lanes_visibility[lane_idx] = valid.astype(np.float32)

            if memory_point_masks is not None:
                memory_points = np.asarray(
                    memory_point_masks[source_lane_idx],
                    dtype=np.float32)[unique_idx]
                memory_interp = np.interp(
                    self.anchor_y_steps, unique_y, memory_points).astype(np.float32)
                gt_lanes_memory_mask[lane_idx] = (
                    valid & (memory_interp > 0.5)).astype(np.float32)

            if (source_lane_idx < len(lanes_uvy)
                    and lanes_uvy[source_lane_idx].shape[0] >= 2):
                uv_y = lanes_uvy[source_lane_idx]
                uv_unique_y, uv_unique_idx = np.unique(uv_y[:, 2], return_index=True)
                if uv_unique_y.shape[0] >= 2:
                    gt_lanes_uv[lane_idx, :, 0] = np.interp(
                        self.anchor_y_steps, uv_unique_y, uv_y[uv_unique_idx, 0])
                    gt_lanes_uv[lane_idx, :, 1] = np.interp(
                        self.anchor_y_steps, uv_unique_y, uv_y[uv_unique_idx, 1])

            ground_lanes[lane_idx, 0:num_y] = xs
            ground_lanes[lane_idx, num_y:2 * num_y] = zs
            ground_lanes[lane_idx, 2 * num_y:3 * num_y] = valid.astype(np.float32)
            ground_lanes[lane_idx, 3 * num_y + category] = 1.0

        return dict(
            gt_lanes_3d=torch.from_numpy(gt_lanes_3d),
            gt_lanes_uv=torch.from_numpy(gt_lanes_uv),
            gt_lanes_visibility=torch.from_numpy(gt_lanes_visibility),
            gt_lanes_category=torch.from_numpy(gt_lanes_category),
            gt_lane_track_ids=torch.from_numpy(gt_lane_track_ids),
            gt_lanes_memory_mask=torch.from_numpy(gt_lanes_memory_mask),
            ground_lanes=torch.from_numpy(ground_lanes),
            raw_lanes_3d=raw_lanes,
            raw_lanes_uv=lanes_uv,
            raw_lanes_uvy=lanes_uvy,
        )

    @staticmethod
    def _split_lane_segments(points, max_gap=3.0):
        """Split a polyline at annotation gaps; never interpolate over one."""
        if points.shape[0] < 2:
            return []
        points = points[np.argsort(points[:, 1])]
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        split_indices = np.where(gaps > max_gap)[0] + 1
        return [segment for segment in np.split(points, split_indices)
                if segment.shape[0] >= 2]

    @staticmethod
    def _arc_length_resample(segments, step=1.0):
        """Resample each continuous lane segment at a fixed arc-length step."""
        output = []
        for segment in segments:
            deltas = np.diff(segment, axis=0)
            lengths = np.linalg.norm(deltas, axis=1)
            arc = np.concatenate(([0.0], np.cumsum(lengths)))
            keep = np.r_[True, np.diff(arc) > 1e-5]
            segment, arc = segment[keep], arc[keep]
            if segment.shape[0] < 2 or arc[-1] < 1e-5:
                continue
            sample_arc = np.r_[np.arange(0.0, arc[-1], step), arc[-1]]
            output.append(np.stack([
                np.interp(sample_arc, arc, segment[:, dim])
                for dim in range(3)], axis=1).astype(np.float32))
        return output

    @staticmethod
    def _kdtree_geometry_metrics(history_points, current_points,
                                 match_radius=0.5):
        """Bidirectional NN metrics in BEV, using KDTree for O(N log N)."""
        history_xy, current_xy = history_points[:, :2], current_points[:, :2]
        if cKDTree is not None:
            d_hc = cKDTree(current_xy).query(history_xy, k=1)[0]
            d_ch = cKDTree(history_xy).query(current_xy, k=1)[0]
        else:
            distances = np.linalg.norm(
                history_xy[:, None] - current_xy[None], axis=-1)
            d_hc, d_ch = distances.min(axis=1), distances.min(axis=0)
        distances = np.concatenate((d_hc, d_ch))
        local = distances[distances <= match_radius]
        return dict(
            chamfer=float(distances.mean()),
            median=float(np.median(distances)),
            p90=float(np.quantile(distances, 0.90)),
            local_chamfer=float(local.mean()) if local.size else float('inf'),
            overlap_current=float((d_ch <= match_radius).mean()),
            overlap_history=float((d_hc <= match_radius).mean()))

    @staticmethod
    def _longitudinal_overlap(history_points, current_points):
        """Return common forward range; heading is intentionally not used."""
        y_min = max(history_points[:, 1].min(), current_points[:, 1].min())
        y_max = min(history_points[:, 1].max(), current_points[:, 1].max())
        return max(0.0, float(y_max - y_min))

    @staticmethod
    def _remove_isolated_lane_outliers(points, jump_threshold=2.0):
        """Remove only spike-like history points; never alter current GT."""
        if points.shape[0] < 3:
            return points
        points = points[np.argsort(points[:, 1])]
        previous = np.linalg.norm(points[1:-1] - points[:-2], axis=1)
        following = np.linalg.norm(points[1:-1] - points[2:], axis=1)
        bridge = np.linalg.norm(points[2:] - points[:-2], axis=1)
        spikes = ((previous > jump_threshold) & (following > jump_threshold)
                  & (bridge < np.minimum(previous, following)))
        keep = np.ones(points.shape[0], dtype=bool)
        keep[1:-1] = ~spikes
        return points[keep]

    def _clip_current_roi_and_fov(self, points, current_calib):
        """Clip a current-ground lane to the requested BEV ROI and camera FOV."""
        point_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        camera = (current_calib['ground_to_norm_cam'] @ point_h.T).T[:, :3]
        projected = (current_calib['K'] @ camera.T).T
        depth = projected[:, 2]
        uv = np.zeros((points.shape[0], 2), dtype=np.float32)
        positive = depth > 1e-5
        uv[positive] = projected[positive, :2] / depth[positive, None]
        in_roi = (
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] <= self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] <= self.point_cloud_range[4]))
        in_fov = (positive & (uv[:, 0] >= 0) & (uv[:, 0] < self.org_w)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < self.org_h))
        return points[in_roi & in_fov]

    def _segments_to_anchors(self, segments):
        """Interpolate only inside observed continuous segments, not gaps."""
        anchors = self.anchor_y_steps
        xz = np.zeros((len(anchors), 2), dtype=np.float32)
        valid = np.zeros(len(anchors), dtype=bool)
        for segment in segments:
            segment = segment[np.argsort(segment[:, 1])]
            ys, indices = np.unique(segment[:, 1], return_index=True)
            segment = segment[indices]
            if len(ys) < 2:
                continue
            start_idx, end_idx = self._anchor_index_bounds(ys[0], ys[-1])
            mask = np.zeros(len(anchors), dtype=bool)
            mask[start_idx:end_idx + 1] = True
            mask &= ~valid
            if np.any(mask):
                xz[mask, 0] = np.interp(anchors[mask], ys, segment[:, 0])
                xz[mask, 1] = np.interp(anchors[mask], ys, segment[:, 2])
                valid[mask] = True
        return xz, valid

    def _anchor_index_bounds(self, y_min, y_max):
        """Quantize lane endpoints to their nearest fixed y-anchor indices.

        The original strict range test dropped a lane endpoint whenever it
        fell between two anchors.  Rounding keeps the nearest anchor (at most
        half an anchor spacing beyond the observed endpoint), preserving edge
        evidence such as y=49.9m in a 50-anchor representation.
        """
        anchors = self.anchor_y_steps
        step = float((anchors[-1] - anchors[0]) / max(len(anchors) - 1, 1))

        def nearest_index(y):
            index = int(np.floor((float(y) - anchors[0]) / step + 0.5))
            return int(np.clip(index, 0, len(anchors) - 1))

        start_idx, end_idx = nearest_index(y_min), nearest_index(y_max)
        return min(start_idx, end_idx), max(start_idx, end_idx)

    def _anchors_inside_support(self, y_min, y_max):
        """Select fixed anchors without endpoint rounding or extrapolation."""
        anchors = self.anchor_y_steps
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return np.zeros(len(anchors), dtype=bool)
        lower, upper = sorted((float(y_min), float(y_max)))
        return (anchors >= lower) & (anchors <= upper)

    def _build_temporal_enhanced_gt(self, label_infos, frame_metas):
        """Reliably supplement current GT gaps with associated history segments.

        Association uses equal track ID, category diagnostics, global-aligned
        KDTree geometry and overlap gates. Current GT is immutable; history
        only fills anchors absent from the original current-frame target.
        """
        current_idx = len(label_infos) - 1
        current_meta = frame_metas[current_idx]
        current_label = label_infos[current_idx]
        current_calib = self._build_calib(current_label)
        current_from_global = np.linalg.inv(
            self._global_from_ground(current_label, current_meta))
        settings = dict(resample_step=1.0, max_segment_gap=3.0,
                        match_radius=0.5, max_local_chamfer=0.25,
                        min_current_coverage=0.60, min_y_overlap=5.0)

        current_records = {
            record['track_id']: record for record in self._parse_lane_records(
                current_label, current_meta)
            if (record['track_id'] >= 0
                and record['ground'].shape[0] >= 2
                and record.get('global', np.empty((0, 3))).shape[0] >= 2)}
        history_records = []
        for frame_idx in range(current_idx):
            for record in self._parse_lane_records(
                    label_infos[frame_idx], frame_metas[frame_idx],
                    is_history=True):
                if (record['track_id'] >= 0
                        and record['ground'].shape[0] >= 2
                        and record.get(
                            'global', np.empty((0, 3))).shape[0] >= 2):
                    history_records.append((frame_idx, record))

        histories_by_id = {}
        for frame_idx, record in history_records:
            histories_by_id.setdefault(record['track_id'], []).append(
                (frame_idx, record))

        # Start from the unmodified official current GT.  This is the key
        # invariant: temporal augmentation may extend but can never shorten
        # or replace any current-frame anchor.
        current_packed = self._parse_openlane_gt(current_label)
        anchors = self.anchor_y_steps
        gt_lanes = current_packed['gt_lanes_3d'].numpy().copy()
        gt_visibility = current_packed['gt_lanes_visibility'].numpy().copy()
        gt_categories = current_packed['gt_lanes_category'].numpy().copy()
        gt_track_ids = current_packed['gt_lane_track_ids'].numpy().copy()
        gt_memory_mask = np.zeros((self.max_lanes, len(anchors)), dtype=np.float32)
        associations = []

        current_lane_indices = {
            int(track_id): lane_idx for lane_idx, track_id in enumerate(gt_track_ids)
            if track_id >= 0}
        for track_id, current_record in current_records.items():
            lane_idx = current_lane_indices.get(track_id)
            if lane_idx is None:
                continue
            current_points = self._clip_current_roi_and_fov(
                current_record['ground'], current_calib)
            current_segments = self._split_lane_segments(
                current_points, settings['max_segment_gap'])
            current_resampled = self._arc_length_resample(
                current_segments, settings['resample_step'])
            current_dense = (
                np.concatenate(current_resampled, axis=0)
                if current_resampled else np.empty((0, 3), dtype=np.float32))
            merged_valid = gt_visibility[lane_idx].astype(bool).copy()
            memory_mask = np.zeros_like(merged_valid, dtype=bool)

            # Newest history is preferred when more than one accepted history
            # can fill the same currently unobserved anchor.
            for frame_idx, history_record in reversed(histories_by_id.get(track_id, [])):
                category_consistent = (
                    history_record['category'] == current_record['category'])
                history_h = np.concatenate([
                    history_record['global'],
                    np.ones((len(history_record['global']), 1), dtype=np.float32)], axis=1)
                history_points = (current_from_global @ history_h.T).T[:, :3].astype(np.float32)
                history_points = self._remove_isolated_lane_outliers(history_points)
                history_points = self._clip_current_roi_and_fov(
                    history_points, current_calib)
                history_segments = self._split_lane_segments(
                    history_points, settings['max_segment_gap'])
                history_resampled = self._arc_length_resample(
                    history_segments, settings['resample_step'])
                history_dense = (
                    np.concatenate(history_resampled, axis=0)
                    if history_resampled else np.empty((0, 3), dtype=np.float32))
                if current_dense.shape[0] < 2 or history_dense.shape[0] < 2:
                    associations.append(dict(track_id=track_id, frame_idx=frame_idx,
                                             accepted=False, reason='insufficient_points'))
                    continue
                metrics = self._kdtree_geometry_metrics(
                    history_dense, current_dense, settings['match_radius'])
                overlap = self._longitudinal_overlap(history_dense, current_dense)
                accepted = (
                    metrics['local_chamfer'] <= settings['max_local_chamfer']
                    and metrics['overlap_current'] >= settings['min_current_coverage']
                    and overlap >= settings['min_y_overlap'])
                associations.append(dict(
                    track_id=track_id, frame_idx=frame_idx, accepted=accepted,
                    category_consistent=category_consistent, y_overlap=overlap,
                    **metrics))
                if not accepted:
                    continue
                history_xz, history_valid = self._segments_to_anchors(history_segments)
                fill = ~merged_valid & history_valid
                gt_lanes[lane_idx, fill, 0] = history_xz[fill, 0]
                gt_lanes[lane_idx, fill, 2] = history_xz[fill, 1]
                merged_valid[fill] = True
                memory_mask[fill] = True

            gt_visibility[lane_idx] = merged_valid.astype(np.float32)
            gt_memory_mask[lane_idx] = memory_mask.astype(np.float32)

        # If a historical frame has at least one independently verified lane,
        # its pose alignment is trusted. Add its other lane IDs that are fully
        # absent from the current annotation as history-only memory lanes.
        trusted_frames = {
            item['frame_idx'] for item in associations if item.get('accepted')}
        added_track_ids = set(int(track_id) for track_id in gt_track_ids if track_id >= 0)
        free_lane_indices = [
            index for index, track_id in enumerate(gt_track_ids) if track_id < 0]
        for frame_idx, history_record in reversed(history_records):
            track_id = history_record['track_id']
            if frame_idx not in trusted_frames or track_id in added_track_ids:
                continue
            if history_record.get('fit_status') == 'linear_fallback':
                associations.append(dict(
                    track_id=track_id, frame_idx=frame_idx, accepted=False,
                    reason='history_only_linear_fallback'))
                continue
            if not free_lane_indices:
                associations.append(dict(
                    track_id=track_id, frame_idx=frame_idx, accepted=False,
                    reason='no_free_query_slot'))
                continue
            history_h = np.concatenate([
                history_record['global'],
                np.ones((len(history_record['global']), 1), dtype=np.float32)], axis=1)
            history_points = (current_from_global @ history_h.T).T[:, :3].astype(np.float32)
            history_points = self._remove_isolated_lane_outliers(history_points)
            history_points = self._clip_current_roi_and_fov(
                history_points, current_calib)
            history_segments = self._split_lane_segments(
                history_points, settings['max_segment_gap'])
            history_xz, history_valid = self._segments_to_anchors(history_segments)
            if history_valid.sum() < 2:
                associations.append(dict(
                    track_id=track_id, frame_idx=frame_idx, accepted=False,
                    reason='history_only_insufficient_anchors'))
                continue
            lane_idx = free_lane_indices.pop(0)
            gt_lanes[lane_idx, :, 0] = history_xz[:, 0]
            gt_lanes[lane_idx, :, 1] = anchors
            gt_lanes[lane_idx, :, 2] = history_xz[:, 1]
            gt_visibility[lane_idx] = history_valid.astype(np.float32)
            gt_categories[lane_idx] = history_record['category']
            gt_track_ids[lane_idx] = track_id
            gt_memory_mask[lane_idx] = history_valid.astype(np.float32)
            added_track_ids.add(track_id)
            associations.append(dict(
                track_id=track_id, frame_idx=frame_idx, accepted=True,
                reason='history_only_merged',
                category=history_record['category'],
                added_anchors=int(history_valid.sum())))

        return dict(
            gt_lanes_3d_enhanced=torch.from_numpy(gt_lanes),
            gt_lanes_visibility_enhanced=torch.from_numpy(gt_visibility),
            gt_lanes_category_enhanced=torch.from_numpy(gt_categories),
            gt_lane_track_ids_enhanced=torch.from_numpy(gt_track_ids),
            gt_lanes_memory_mask=torch.from_numpy(gt_memory_mask),
            temporal_gt_associations=associations,
        )

    def _global_from_ground(self, label_info, frame_meta):
        calib = self._build_calib(label_info)
        global_from_vehicle = np.eye(4, dtype=np.float32)
        global_from_vehicle[:3, :3] = np.asarray(
            frame_meta['ego2global_rotation'], dtype=np.float32)
        global_from_vehicle[:3, 3] = np.asarray(
            frame_meta['ego2global_translation'], dtype=np.float32)
        return (global_from_vehicle @ calib['cam_to_vehicle'] @ np.linalg.inv(
            calib['lane_cam_to_ground'])).astype(np.float32)
    # ----------------------------------------------------gt build end----------------------------------------------------

    # ---------------------------------------------------lidar load start---------------------------------------------------
    def _load_lidar_points(self, info, current_pose, queue_index):
        lidar_points = np.load(info['lidar_file'])
        if lidar_points.ndim != 2 or lidar_points.shape[1] < 5:
            return np.zeros((0, self.num_lidar_feat), dtype=np.float32)

        if lidar_points.shape[1] >= 6:
            valid_mask = lidar_points[:, 5] <= 0
            if np.any(valid_mask):
                lidar_points = lidar_points[valid_mask]

        xyz = lidar_points[:, :3].astype(np.float32)
        feature = lidar_points[:, 3:5].astype(np.float32)

        # Accumulated sweeps enter the LiDAR backbone together, so every point
        # must first be represented in the current ego frame.  Poses are
        # ego-to-global transforms.
        history_pose = np.asarray(info['pose'], dtype=np.float32).reshape(4, 4)
        history_to_current = (
            np.linalg.inv(current_pose) @ history_pose).astype(np.float32)
        xyz = (history_to_current[:3, :3] @ xyz.T).T
        xyz = xyz + history_to_current[:3, 3]
        xyz = (WAYMO_VEHICLE_TO_GROUND_AXES @ xyz.T).T

        queue_column = np.full((xyz.shape[0], 1), float(queue_index), dtype=np.float32)
        return np.concatenate([xyz, feature, queue_column], axis=1)

    def _filter_points_by_range(self, points):
        if self.point_cloud_range is None or points.shape[0] == 0:
            return points
        pc_range = self.point_cloud_range
        mask = (
            (points[:, 0] >= pc_range[0])
            & (points[:, 1] >= pc_range[1])
            & (points[:, 2] >= pc_range[2])
            & (points[:, 0] <= pc_range[3])
            & (points[:, 1] <= pc_range[4])
            & (points[:, 2] <= pc_range[5])
        )
        return points[mask]

    def _process_points(self, points):
        points = self._filter_points_by_range(points)
        if self.shuffle_points:
            np.random.shuffle(points)
        return points.astype(np.float32)

    def _load_lidar_queue(self, lidar_scene, queue_keys):
        if lidar_scene is None or queue_keys[-1] not in lidar_scene['frames']:
            return torch.zeros((0, self.num_lidar_feat), dtype=torch.float32)

        current_pose = lidar_scene['frames'][queue_keys[-1]]['pose']
        lidar_frames = []
        # Distance/neighbor padding can repeat the current frame.  Repeating
        # identical points would only inflate voxel density, so keep the last
        # occurrence of each physical sweep.
        last_occurrence = {
            key: index for index, key in enumerate(queue_keys)}
        for queue_index, key in enumerate(queue_keys):
            if last_occurrence[key] != queue_index:
                continue
            info = lidar_scene['frames'].get(key)
            if info is None:
                points = np.zeros((0, self.num_lidar_feat), dtype=np.float32)
            else:
                points = self._load_lidar_points(info, current_pose, queue_index)
            points = self._process_points(points)
            if points.shape[0] > 0:
                lidar_frames.append(points)
        if not lidar_frames:
            return torch.zeros((0, self.num_lidar_feat), dtype=torch.float32)
        return torch.from_numpy(np.concatenate(lidar_frames, axis=0))
    # ----------------------------------------------------lidar load end----------------------------------------------------

    # ---------------------------------------------------getitem start---------------------------------------------------
    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        current_json_path = self.label_files[idx]
        scene_name = current_json_path.parent.name
        scenario_name = self._scenario_name(current_json_path)
        current_key = _timestamp_key(current_json_path)
        label_scene = self._load_label_scene(scene_name, current_json_path.parent)

        queue_keys, sampling_mode, lidar_scene, label_scene = self._sample_frame_queue_keys(
            scene_name, current_key, label_scene=label_scene)

        img_tensors = []
        view_masks = []
        label_infos = []
        frame_metas = {}
        random_aug_replays = (
            [None] * self.num_views
            if self.view_mode == 'random_aug' and self.img_aug_replay is not None
            else None
        )
        view_aug_seeds = (
            np.random.randint(0, 2**31 - 1, size=self.num_views).tolist()
            if self.view_mode == 'multi_aug_views'
            else None
        )
        current_label_info = self._read_json(current_json_path)
        current_pose = np.asarray(
            current_label_info.get('pose', np.eye(4)),
            dtype=np.float32).reshape(4, 4)
        if lidar_scene is not None and current_key in lidar_scene['frames']:
            current_pose = lidar_scene['frames'][current_key]['pose']
        for key in queue_keys:
            json_path = label_scene['frames'].get(key)
            if json_path is None:
                raise KeyError(
                    f'Missing label/image metadata for sampled frame '
                    f'{scene_name}/{key}.')
            label_info = self._read_json(json_path)
            image_path = self._find_image_path(label_info, scene_name, key)
            image_views, active_view_mask = self._load_image_views(
                image_path,
                random_aug_replays=random_aug_replays,
                view_aug_seeds=view_aug_seeds)
            img_tensors.append(image_views)
            view_masks.append(active_view_mask)
            label_infos.append(label_info)
            queue_index = len(label_infos) - 1
            frame_metas[queue_index] = self._build_frame_meta(
                scene_name, key, label_info, image_path, lidar_scene, current_pose)

        # 多增强视图构造: OpenLane 当前只有前视相机, 这里将同一图像构造成 num_views 个增强视图。
        # 后续模型侧可基于 view_mask 做增强视图 dropout、特征融合和 Image-LiDAR 鲁棒融合。
        img = torch.stack(img_tensors, dim=0)
        if self.grid_mask is not None:
            image_shape = img.shape
            img = self.grid_mask(
                img.reshape(-1, *image_shape[-3:])).reshape(image_shape)
        view_mask = torch.stack(view_masks, dim=0)
        lidar = self._load_lidar_queue(lidar_scene, queue_keys)
        gt = self._parse_openlane_gt(label_infos[-1])
        if self.temporal_gt_augmentation:
            gt.update(self._build_temporal_enhanced_gt(label_infos, frame_metas))

        sample = dict(
            img=img,
            lidar=lidar,
            img_metas=dict(
                scene_name=scene_name,
                scenario_name=scenario_name,
                frame_keys=queue_keys,
                current_key=current_key,
                sampling_mode=sampling_mode,
                num_history_frames=len(queue_keys) - 1,
                view_mode=self.view_mode,
                num_views=self.num_views,
                view_names=self.view_aug_names[:self.num_views],
                view_mask=view_mask,
                queue=frame_metas,
                current=frame_metas[len(queue_keys) - 1],
            ),
            **gt,
            raw_label=label_infos[-1],
        )
        if not self.pack_keys:
            return sample
        return {key: sample[key] for key in self.pack_keys if key in sample}
    # ----------------------------------------------------getitem end----------------------------------------------------


@FUNCTIONS.register_module()
def openlane_collate(batch):
    """Collate fixed-shape images and keep variable-length points per sample.

    Raw points are
    not stacked to a fixed N in the dataloader.
    """
    collated = dict(
        img=torch.stack([item['img'] for item in batch], dim=0),
        lidar=[item['lidar'] for item in batch],
        img_metas=[item['img_metas'] for item in batch],
        view_mask=torch.stack([item['img_metas']['view_mask'] for item in batch], dim=0),
        raw_label=[item['raw_label'] for item in batch],
    )
    for key in (
        'gt_lanes_3d',
        'gt_lanes_uv',
        'gt_lanes_visibility',
        'gt_lanes_category',
        'gt_lane_track_ids',
        'gt_lanes_memory_mask',
        'gt_lanes_3d_enhanced',
        'gt_lanes_visibility_enhanced',
        'gt_lanes_category_enhanced',
        'gt_lane_track_ids_enhanced',
        'ground_lanes',
    ):
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in ('raw_lanes_3d', 'raw_lanes_uv', 'raw_lanes_uvy'):
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]
    return collated
