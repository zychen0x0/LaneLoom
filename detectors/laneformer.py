from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.ops import Voxelization
from mmengine.model import BaseModel
from mmdet3d.registry import MODELS

from LaneFormer.augmentations import GridMask


def _build_module(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    return MODELS.build(cfg)


def _build_voxel_layer(cfg):
    if cfg is None or isinstance(cfg, nn.Module):
        return cfg
    return Voxelization(**cfg)


@MODELS.register_module()
class LaneFormer(BaseModel):
    """LaneFormer image/LiDAR detector with a temporal BEV queue."""

    # ---------------------------------------------------init start---------------------------------------------------
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 lane_head=None,
                 num_levels=None,
                 frames=(0,),
                 pts_input_dim=5,
                 freeze_img=False,
                 data_preprocessor=None,
                 ):
        super().__init__(data_preprocessor=data_preprocessor)

        self.pts_voxel_layer = _build_voxel_layer(pts_voxel_layer)
        self.pts_voxel_encoder = _build_module(pts_voxel_encoder)
        self.pts_middle_encoder = _build_module(pts_middle_encoder)
        self.pts_backbone = _build_module(pts_backbone)
        self.pts_neck = _build_module(pts_neck)

        self.img_backbone = _build_module(img_backbone)
        self.img_neck = _build_module(img_neck)
        self.lane_head = _build_module(lane_head)

        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.num_levels = num_levels
        self.frames = tuple(frames)
        self.pts_input_dim = pts_input_dim

        self.freeze_img = freeze_img
        self._freeze_img_branch()
    # ----------------------------------------------------init end----------------------------------------------------

    # ---------------------------------------------------init weights start---------------------------------------------------
    def init_weights(self):
        """Initialize submodules and optionally freeze the image branch."""
        init_pairs = [
            (self.img_backbone, None),
            (self.img_neck, None),
            (self.pts_voxel_encoder, None),
            (self.pts_middle_encoder, None),
            (self.pts_backbone, None),
            (self.pts_neck, None),
            (self.lane_head, None),
        ]
        for module, ckpt in init_pairs:
            if module is not None and hasattr(module, 'init_weights'):
                module.init_weights(ckpt) if ckpt is not None else module.init_weights()

        self._freeze_img_branch()

    def _freeze_img_branch(self):
        if not self.freeze_img:
            return
        for module in (self.img_backbone, self.img_neck):
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    # ----------------------------------------------------init weights end----------------------------------------------------

    # ---------------------------------------------------properties start---------------------------------------------------
    @property
    def with_img_backbone(self):
        return self.img_backbone is not None

    @property
    def with_img_neck(self):
        return self.img_neck is not None

    @property
    def with_pts_backbone(self):
        return (
            self.pts_voxel_layer is not None
            and self.pts_voxel_encoder is not None
            and self.pts_middle_encoder is not None
            and self.pts_backbone is not None
        )

    @property
    def with_pts_neck(self):
        return self.pts_neck is not None

    @property
    def with_pts_branch(self):
        return self.with_pts_backbone

    # ----------------------------------------------------properties end----------------------------------------------------

    # ---------------------------------------------------image feature start---------------------------------------------------
    def extract_img_feat(self, img, img_metas=None):
        """Extract multi-level image features.

        Args:
            img: Tensor with shape [B, N, C, H, W] for one time step.
            img_metas: Meta information of each sample.
        Returns:
            list[Tensor]: each level has shape [B, N, C, H, W].
        """
        if not self.with_img_backbone or img is None:
            return None

        input_shape = img.shape[-2:]
        if img_metas is not None:
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

        num_views = 1
        batch_size = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            num_views = img.size(1)
            img = img.squeeze(0)
        elif img.dim() == 5 and img.size(0) > 1:
            batch_size, num_cams, channels, height, width = img.size()
            num_views = num_cams
            img = img.reshape(batch_size * num_cams, channels, height, width)

        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img.float())
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if not isinstance(img_feats, (list, tuple)):
            img_feats = [img_feats]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            batch_cam, channels, height, width = img_feat.size()
            num_cams = int(batch_cam / batch_size) if batch_size > 0 else num_views
            img_feats_reshaped.append(
                img_feat.view(batch_size, num_cams, channels, height, width))
        return img_feats_reshaped

    def extract_feat(self, img=None, points=None, img_metas=None):
        """Extract image and optional LiDAR features."""
        img_feats = self.extract_img_feat(
            img, img_metas=img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return img_feats, pts_feats
    # ----------------------------------------------------image feature end----------------------------------------------------

    # ---------------------------------------------------lidar feature start---------------------------------------------------
    def voxelize(self, points):
        """Apply hard voxelization to a batch of point clouds."""
        if self.pts_voxel_layer is None:
            return points
        voxels, coors, num_points = [], [], []
        for batch_idx, sample_points in enumerate(points):
            if sample_points is None or sample_points.numel() == 0:
                continue
            sample_points = sample_points[:, :self.pts_input_dim].contiguous()
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(
                sample_points)
            voxels.append(res_voxels)
            coors.append(
                torch.nn.functional.pad(
                    res_coors, (1, 0), mode='constant', value=batch_idx))
            num_points.append(res_num_points)

        if not voxels:
            return None, None, None
        return (
            torch.cat(voxels, dim=0),
            torch.cat(num_points, dim=0),
            torch.cat(coors, dim=0))

    def extract_pts_feat(self, points, img_feats=None, img_metas=None):
        """Extract LiDAR BEV features from variable-length points."""
        if points is None or not self.with_pts_backbone:
            return None
        if not isinstance(points, (list, tuple)):
            points = [points]

        voxels, num_points, coors = self.voxelize(points)
        if voxels is None:
            return None

        voxel_features = self.pts_voxel_encoder(
            voxels,
            num_points,
            coors)
        batch_size = len(points)
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    # ----------------------------------------------------lidar feature end----------------------------------------------------

    def obtain_history_bev(self, img_queue, img_metas_queue=None,
                           points_queue=None):
        """Obtain image-only history BEVs iteratively without gradients."""
        is_training = self.training
        self.eval()

        prev_bev = OrderedDict((frame, None) for frame in self.frames if frame < 0)
        with torch.no_grad():
            for queue_idx, frame in enumerate([frame for frame in self.frames if frame < 0]):
                img = img_queue[:, queue_idx, ...]
                img_metas = self._select_queue_metas(
                    img_metas_queue, queue_idx, frame)
                img_feats = self.extract_img_feat(img, img_metas=img_metas)
                if self.num_levels is not None and img_feats is not None:
                    img_feats = img_feats[:self.num_levels]

                bev = self.lane_head.forward_image_bev(
                    img_feats=img_feats,
                    img_metas=img_metas)
                # Keep frame calibration with BEV; the head pose-warps it to current ego.
                prev_bev[frame] = None if bev is None else dict(
                    bev=bev.detach(), img_metas=img_metas)

        if is_training:
            self.train()
        return list(prev_bev.values())

    def _encode_batch(self, batch):
        """Encode the current frame and pose-aligned history from one OpenLane batch."""
        img = batch['img']
        if img.ndim != 6:
            raise ValueError(
                f'Expected img shape [B, T, N, C, H, W], got {tuple(img.shape)}.')

        points = batch.get('lidar', batch.get('points'))
        img_metas = batch['img_metas']
        queue_length = img.size(1)
        prev_bev = None
        if queue_length > 1:
            prev_bev = self.obtain_history_bev(
                img[:, :-1, ...], img_metas_queue=img_metas, points_queue=points)

        current_metas = self._select_queue_metas(img_metas, queue_length - 1, 0)
        # Dataset sweeps have already been pose-aligned to current ego.  Feed
        # the whole accumulated cloud through the LiDAR backbone exactly once.
        current_points = points
        if torch.is_tensor(current_points):
            current_points = [current_points]
        img_feats, pts_feats = self.extract_feat(
            img=img[:, -1, ...], points=current_points, img_metas=current_metas)
        if self.num_levels is not None and img_feats is not None:
            img_feats = img_feats[:self.num_levels]
        return img_feats, pts_feats, current_metas, current_points, prev_bev

    def loss(self, batch):
        (img_feats, pts_feats, img_metas, points,
         prev_bev) = self._encode_batch(batch)
        return self.lane_head.loss(
            img_feats=img_feats,
            pts_feats=pts_feats,
            img_metas=img_metas,
            points=points,
            prev_bev=prev_bev,
            **self._lane_targets(batch))

    def predict(self, batch):
        (img_feats, pts_feats, img_metas, points,
         prev_bev) = self._encode_batch(batch)
        return self.lane_head.predict(
            img_feats=img_feats,
            pts_feats=pts_feats,
            img_metas=img_metas,
            points=points,
            prev_bev=prev_bev)

    @staticmethod
    def _lane_targets(batch):
        target_keys = (
            'gt_lanes_3d', 'gt_lanes_uv', 'gt_lanes_visibility',
            'gt_lanes_category', 'gt_lane_track_ids', 'gt_lanes_memory_mask',
            'gt_lanes_3d_enhanced', 'gt_lanes_visibility_enhanced',
            'gt_lanes_category_enhanced',
            'gt_lane_track_ids_enhanced', 'ground_lanes', 'raw_lanes_3d',
            'raw_lanes_uv', 'raw_lanes_uvy')
        return {key: batch[key] for key in target_keys if key in batch}

    def forward(self, mode='tensor', **batch):
        """Run one OpenLane batch using MMEngine's keyword-batch convention."""
        if not batch:
            raise ValueError('LaneFormer expects an OpenLane batch dictionary.')
        if mode == 'loss':
            return self.loss(batch)
        if mode in ('predict', 'tensor'):
            return self.predict(batch)
        raise ValueError(f'Unsupported MMEngine forward mode: {mode}')

    def train_step(self, data, optim_wrapper):
        batch = self.data_preprocessor(data, True)
        losses = self.forward(mode='loss', **batch)
        loss, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(loss)
        return log_vars

    def val_step(self, data):
        batch = self.data_preprocessor(data, False)
        predictions = self.forward(mode='predict', **batch)
        return self._build_evaluation_samples(predictions, batch)

    def test_step(self, data):
        return self.val_step(data)

    @staticmethod
    def _build_evaluation_samples(predictions, batch):
        """Attach current-frame GT and metadata to one prediction per sample.

        MMEngine metrics consume a sequence, while the lane head predicts an
        entire batch in one tensor dictionary.  Both current-frame and
        optional temporal-supplement targets are exposed so the evaluator
        configuration can use the same supervision target as training.
        """
        samples = []
        metas = batch.get('img_metas', [{} for _ in range(predictions['x'].size(0))])
        for index in range(predictions['x'].size(0)):
            meta = metas[index] if isinstance(metas, (list, tuple)) else metas
            samples.append(dict(
                pred_x=predictions['x'][index],
                pred_z=predictions['z'][index],
                pred_visibility=predictions['visibility'][index],
                pred_scores=predictions['scores'][index],
                pred_category=predictions['category'][index],
                anchor_y_steps=predictions['anchor_y_steps'],
                gt_lanes_3d=batch['gt_lanes_3d'][index],
                gt_lanes_visibility=batch['gt_lanes_visibility'][index],
                gt_lanes_category=batch['gt_lanes_category'][index],
                gt_lanes_3d_enhanced=(
                    batch['gt_lanes_3d_enhanced'][index]
                    if 'gt_lanes_3d_enhanced' in batch else None),
                gt_lanes_visibility_enhanced=(
                    batch['gt_lanes_visibility_enhanced'][index]
                    if 'gt_lanes_visibility_enhanced' in batch else None),
                gt_lanes_category_enhanced=(
                    batch['gt_lanes_category_enhanced'][index]
                    if 'gt_lanes_category_enhanced' in batch else None),
                scene_name=meta.get('scene_name', ''),
                scenario_name=meta.get('scenario_name', ''),
                frame_key=meta.get('current_key', ''),
                road_curvature=(
                    predictions['road_curvature'][index]
                    if 'road_curvature' in predictions else None),
                road_grade=(
                    predictions['road_grade'][index]
                    if 'road_grade' in predictions else None),
                initial_anchor_points=(
                    predictions['anchor_points'][index]
                    if 'anchor_points' in predictions else None),
            ))
        return samples

    # ---------------------------------------------------queue utils start---------------------------------------------------
    def _select_queue_metas(self, img_metas, queue_idx, frame_key):
        if img_metas is None:
            return None
        if isinstance(img_metas, dict):
            return [img_metas]
        if isinstance(img_metas, (list, tuple)):
            # LaneFormer dataloader returns one meta dict per batch sample.
            if img_metas and isinstance(img_metas[0], dict):
                selected = []
                for meta in img_metas:
                    meta = dict(meta)
                    queue = meta.get('queue')
                    if isinstance(queue, dict) and queue:
                        frame_meta = queue.get(queue_idx, queue.get(frame_key))
                        meta['current'] = frame_meta or queue[max(queue)]
                    selected.append(meta)
                return selected
            # Accept nested metadata lists produced by generic data loaders.
            if img_metas and isinstance(img_metas[0], (list, tuple)):
                return [meta[queue_idx] if len(meta) > queue_idx else meta[-1]
                        for meta in img_metas]
        return img_metas

    def _select_queue_points(self, points, queue_idx):
        if points is None:
            return None
        selected = []
        for sample_points in points:
            if not torch.is_tensor(sample_points) or sample_points.numel() == 0:
                selected.append(sample_points)
                continue
            selected.append(sample_points[sample_points[:, -1] == queue_idx])
        return selected
    # ----------------------------------------------------queue utils end----------------------------------------------------
