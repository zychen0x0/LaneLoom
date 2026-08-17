"""ONCE-3DLanes image/LiDAR dataset for LaneLoom."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from mmengine.registry import FUNCTIONS
from mmdet3d.registry import DATASETS

from ..openlane.dataset import OpenLaneDataset, openlane_collate


CAMERA_TO_GROUND = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

_PIPELINE_TYPE_MAP = {
    'LoadONCEImages': 'LoadTemporalOpenLaneImages',
    'ResizeONCEImages': 'ResizeOpenLaneImages',
    'NormalizeONCEImages': 'NormalizeOpenLaneImages',
    'LoadONCEPoints': 'LoadPoseAlignedOpenLanePoints',
    'LoadONCEAnnotations': 'LoadOpenLaneAnnotations',
}


def _pose_matrix(pose):
    """Convert ONCE (qx, qy, qz, qw, tx, ty, tz) into a rigid transform."""
    qx, qy, qz, qw, tx, ty, tz = (float(value) for value in pose)
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-8:
        rotation = np.eye(3, dtype=np.float32)
    else:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        rotation = np.array([
            [1 - 2 * (qy * qy + qz * qz),
             2 * (qx * qy - qz * qw),
             2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw),
             1 - 2 * (qx * qx + qz * qz),
             2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw),
             2 * (qy * qz + qx * qw),
             1 - 2 * (qx * qx + qy * qy)],
        ], dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (tx, ty, tz)
    return matrix


@DATASETS.register_module()
class ONCEDataset(OpenLaneDataset):
    """Read ONCE lanes in camera axes and LiDAR in the same ground frame.

    ONCE camera points ``(x, y, z)`` become LaneLoom ground points
    ``(x, z, -y)``.  Raw LiDAR points are transformed through the official
    ``cam_to_velo`` calibration before entering that ground frame.

    This first ONCE integration is deliberately single-frame.  ONCE lane
    annotations do not carry the track IDs required by LaneLoom's temporal-GT
    supplementation, so enabling that OpenLane-specific supervision would
    change the target set rather than merely add temporal inputs.
    """

    METAINFO = dict(dataset_name='ONCE-3DLanes')

    def __init__(self, dataset_cfg, temporal_sampling_distance,
                 smooth_3d_gt, gt_outlier_residual, modality,
                 test_mode=False, split=None, max_samples=None):
        if temporal_sampling_distance is not False:
            raise ValueError(
                'ONCEDataset currently requires '
                'temporal_sampling_distance=False; ONCE lane labels have no '
                'track IDs for temporal GT augmentation.')

        original_cfg = dict(dataset_cfg)
        self.info_root = Path(original_cfg['info_dir']).expanduser().resolve()
        self.split_root = (
            Path(original_cfg['split_dir']).expanduser().resolve())

        parent_cfg = dict(original_cfg)
        for pipeline_name in ('train_pipeline', 'val_pipeline'):
            parent_cfg[pipeline_name] = [
                dict(stage, type=_PIPELINE_TYPE_MAP.get(
                    stage.get('type'), stage.get('type')))
                for stage in original_cfg[pipeline_name]
            ]
        super().__init__(
            dataset_cfg=parent_cfg,
            temporal_sampling_distance=False,
            smooth_3d_gt=smooth_3d_gt,
            gt_outlier_residual=gt_outlier_residual,
            modality=modality,
            test_mode=test_mode,
            split=split,
            max_samples=max_samples)
        self.dataset_cfg = original_cfg

    # Index construction.
    @staticmethod
    def _frame_key(path):
        return Path(path).stem

    @staticmethod
    def _sequence_name(label_path):
        # ONCE_3DLanes/{train,val}/<sequence>/cam01/<frame>.json
        return Path(label_path).parents[1].name

    def _annotation_split(self):
        if self.split.startswith('train'):
            return 'train'
        if self.split == self.dataset_cfg.get('val_split', 'val'):
            return 'val'
        raise ValueError(
            f'ONCE split {self.split!r} has no public ground truth; '
            'local quantitative evaluation must use val.')

    def _load_label_files(self, max_samples=None):
        split_file = self.split_root / f'{self.split}.txt'
        if not split_file.is_file():
            split_file = self.label_root / 'list' / f'{self.split}.txt'
        if not split_file.is_file():
            raise FileNotFoundError(
                f'ONCE split file does not exist: {split_file}')

        annotation_split = self._annotation_split()
        label_files = []
        with open(split_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                relative = line.strip().lstrip('/')
                if not relative:
                    continue
                label_files.append(
                    self.label_root / annotation_split
                    / Path(relative).with_suffix('.json'))
        if max_samples is not None:
            label_files = label_files[:int(max_samples)]
        label_files = label_files[::self.load_interval]
        if not label_files:
            raise FileNotFoundError(f'No ONCE samples listed in {split_file}')
        return label_files

    def _load_lidar_segment_dirs(self):
        return {}

    def _load_label_scene(self, scene_name, scene_dir=None):
        scene_dir = Path(scene_dir or (
            self.label_root / self._annotation_split() / scene_name / 'cam01'))
        cache_key = str(scene_dir)
        if cache_key in self._label_scene_cache:
            return self._label_scene_cache[cache_key]
        frames = {
            path.stem: path for path in sorted(scene_dir.glob('*.json'))}
        scene = dict(keys=sorted(frames), frames=frames)
        self._label_scene_cache[cache_key] = scene
        return scene

    def _load_lidar_scene(self, scene_name):
        if scene_name in self._lidar_scene_cache:
            return self._lidar_scene_cache[scene_name]

        info_path = self.info_root / scene_name / f'{scene_name}.json'
        if not info_path.is_file():
            self._lidar_scene_cache[scene_name] = None
            return None
        with open(info_path, 'r', encoding='utf-8') as handle:
            sequence_info = json.load(handle)

        cam_to_velo = np.asarray(
            sequence_info['calib']['cam01']['cam_to_velo'],
            dtype=np.float32).reshape(4, 4)
        lidar_to_ground = (
            CAMERA_TO_GROUND @ np.linalg.inv(cam_to_velo)).astype(np.float32)
        frames = {}
        for frame in sequence_info.get('frames', []):
            frame_key = str(frame['frame_id'])
            lidar_file = (
                self.lidar_root / scene_name / 'lidar_roof'
                / f'{frame_key}.bin')
            if not lidar_file.is_file() or frame.get('pose') is None:
                continue
            frames[frame_key] = dict(
                lidar_file=lidar_file,
                pose=_pose_matrix(frame['pose']),
                timestamp_micros=int(frame_key),
                lidar_to_ground=lidar_to_ground,
            )
        scene = dict(keys=sorted(frames), frames=frames)
        self._lidar_scene_cache[scene_name] = scene
        return scene

    def _current_frame_context(self, index):
        current_path = self.label_files[index]
        scene_name = self._sequence_name(current_path)
        current_key = self._frame_key(current_path)
        lidar_scene = self._load_lidar_scene(scene_name)
        current_pose = (
            lidar_scene['frames'][current_key]['pose']
            if lidar_scene is not None and current_key in lidar_scene['frames']
            else np.eye(4, dtype=np.float32))
        return current_path, scene_name, current_key, lidar_scene, current_pose
    # End index construction.

    # Calibration and ground truth.
    def _build_calib(self, label_info):
        projection = np.asarray(
            label_info['calibration'], dtype=np.float32).reshape(3, 4)
        camera_intrinsics = projection[:, :3]
        ground_to_camera = np.linalg.inv(CAMERA_TO_GROUND).astype(np.float32)
        ground_to_raw_img = (
            camera_intrinsics @ ground_to_camera[:3]).astype(np.float32)
        return dict(
            K=camera_intrinsics,
            cam_to_vehicle=CAMERA_TO_GROUND.copy(),
            vehicle_to_cam=ground_to_camera.copy(),
            lane_cam_to_ground=CAMERA_TO_GROUND.copy(),
            norm_cam_to_ground=CAMERA_TO_GROUND.copy(),
            norm_cam_to_vehicle=CAMERA_TO_GROUND.copy(),
            vehicle_to_norm_cam=ground_to_camera.copy(),
            vehicle_to_ground=np.eye(4, dtype=np.float32),
            ground_to_norm_cam=ground_to_camera.copy(),
            vehicle_to_raw_img=ground_to_raw_img.copy(),
            ground_to_raw_img=ground_to_raw_img,
        )

    def _find_image_path(self, label_info, scene_name, frame_key):
        image_path = (
            self.image_root / scene_name / 'cam01' / f'{frame_key}.jpg')
        if not image_path.is_file():
            raise FileNotFoundError(f'No ONCE image found: {image_path}')
        return image_path

    def _parse_lane_records(self, label_info, frame_meta=None,
                            is_history=False):
        calib = self._build_calib(label_info)
        image_aug = self._image_resize_matrix()
        records = []
        for lane_index, points in enumerate(label_info.get('lanes', [])):
            lane_camera = np.asarray(points, dtype=np.float32)
            if lane_camera.ndim != 2 or lane_camera.shape[0] < 2 \
                    or lane_camera.shape[1] != 3:
                continue
            uv = self._project_lane_cam_to_uv(lane_camera, calib['K'])
            lane_h = np.concatenate([
                lane_camera,
                np.ones((lane_camera.shape[0], 1), dtype=np.float32),
            ], axis=1)
            lane_ground = (
                CAMERA_TO_GROUND @ lane_h.T).T[:, :3].astype(np.float32)
            order = np.argsort(lane_ground[:, 1])
            lane_ground = lane_ground[order]
            uv = uv[order]
            lane_ground_raw = lane_ground.copy()
            lane_ground, smooth_keep, fit_status = self._smooth_lane_ground(
                lane_ground)
            uv = uv[smooth_keep]
            uv_ground_y = lane_ground_raw[smooth_keep, 1:2]
            uv_h = np.concatenate([
                uv, np.ones((uv.shape[0], 1), dtype=np.float32),
            ], axis=1)
            uv_net = (uv_h @ image_aug.T)[:, :2].astype(np.float32)
            uvy = np.concatenate([uv_net, uv_ground_y], axis=1)
            records.append(dict(
                ground=lane_ground,
                ground_raw=lane_ground_raw,
                uv=uv_net,
                uvy=uvy,
                category=0,
                track_id=-1,
                fit_status=fit_status,
                is_history=bool(is_history),
                source_lane_index=lane_index,
            ))
        return records

    def _parse_once_gt(self, label_info):
        return super()._parse_openlane_gt(label_info)
    # End calibration and ground truth.

    # LiDAR loading.
    def _load_lidar_points(self, info, current_pose, queue_index):
        del queue_index
        raw = np.fromfile(info['lidar_file'], dtype=np.float32)
        if raw.size == 0 or raw.size % 4:
            return np.zeros((0, self.num_lidar_feat), dtype=np.float32)
        raw = raw.reshape(-1, 4)
        xyz = raw[:, :3]
        history_pose = np.asarray(
            info['pose'], dtype=np.float32).reshape(4, 4)
        history_to_current = (
            np.linalg.inv(current_pose) @ history_pose).astype(np.float32)
        xyz_h = np.concatenate([
            xyz, np.ones((xyz.shape[0], 1), dtype=np.float32),
        ], axis=1)
        current_xyz = (history_to_current @ xyz_h.T).T
        ground_xyz = (
            np.asarray(info['lidar_to_ground'], dtype=np.float32)
            @ current_xyz.T).T[:, :3]
        return np.concatenate(
            [ground_xyz, raw[:, 3:4]], axis=1
        ).astype(np.float32)
    # End LiDAR loading.

    # Sample construction.
    def get_data_info(self, index):
        (current_path, scene_name, current_key,
         lidar_scene, current_pose) = self._current_frame_context(index)
        label_info = self._read_json(current_path)
        image_path = self._find_image_path(label_info, scene_name, current_key)
        frame_meta = self._build_frame_meta(
            scene_name, current_key, label_info, image_path,
            lidar_scene, current_pose)
        input_dict = self.prepare_input_dict(frame_meta)
        if not self.test_mode:
            input_dict['ann_info'] = self._parse_once_gt(label_info)
        return input_dict

    def __getitem__(self, idx):
        (current_path, scene_name, current_key,
         lidar_scene, current_pose) = self._current_frame_context(idx)
        label_scene = self._load_label_scene(scene_name, current_path.parent)
        queue_keys, sampling_mode, _, label_scene = (
            self._sample_frame_queue_keys(
                scene_name, current_key, label_scene=label_scene))
        img_tensors = []
        view_masks = []
        label_infos = []
        frame_metas = {}
        recipe_indices = self._sample_augmentation_recipe_indices()
        recipe_seeds = (
            np.random.randint(0, 2**31 - 1, size=self.num_views).tolist()
            if self.image_augmentation_mode == 'named_recipes' else None)
        for queue_index, key in enumerate(queue_keys):
            json_path = label_scene['frames'].get(key)
            if json_path is None:
                raise KeyError(f'Missing ONCE label for {scene_name}/{key}.')
            label_info = self._read_json(json_path)
            image_path = self._find_image_path(label_info, scene_name, key)
            image_views, active_view_mask = self._load_image_views(
                image_path,
                augmentation_recipe_indices=recipe_indices,
                augmentation_recipe_seeds=recipe_seeds)
            img_tensors.append(image_views)
            view_masks.append(active_view_mask)
            label_infos.append(label_info)
            frame_metas[queue_index] = self._build_frame_meta(
                scene_name, key, label_info, image_path,
                lidar_scene, current_pose)

        img = torch.stack(img_tensors, dim=0)
        if self.grid_mask is not None:
            image_shape = img.shape
            img = self.grid_mask(
                img.reshape(-1, *image_shape[-3:])).reshape(image_shape)
        view_mask = torch.stack(view_masks, dim=0)
        sample = dict(
            img=img,
            lidar=self._load_lidar_queue(lidar_scene, queue_keys),
            img_metas=dict(
                scene_name=scene_name,
                scenario_name='',
                frame_keys=queue_keys,
                current_key=current_key,
                sampling_mode=sampling_mode,
                num_history_frames=0,
                image_augmentation_mode=self.image_augmentation_mode,
                num_views=self.num_views,
                augmentation_recipe_names=[
                    self.augmentation_recipe_names[index]
                    for index in (recipe_indices or [])
                    if index < len(self.augmentation_recipe_names)
                ],
                view_mask=view_mask,
                queue=frame_metas,
                current=frame_metas[len(queue_keys) - 1],
            ),
            **self._parse_once_gt(label_infos[-1]),
            raw_label=label_infos[-1],
        )
        if not self.pack_keys:
            return sample
        return {
            key: sample[key] for key in self.pack_keys if key in sample
        }
    # End sample construction.


@FUNCTIONS.register_module()
def once_collate(batch):
    """Collate ONCE samples while preserving variable-length LiDAR points."""
    return openlane_collate(batch)
