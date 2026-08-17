"""ONCE-3DLanes image + LiDAR dataset configuration."""

import os


dataset_type = 'ONCEDataset'
dataset_name = 'once'

laneloom_root = os.path.abspath(
    os.path.join('{{ fileDirname }}', '..', '..', '..'))
local_data_root = os.path.join(laneloom_root, 'data')
data_root = os.environ.get('LANELOOM_DATA_ROOT', local_data_root)
once_root = os.path.join(data_root, 'once_3dlanes')

# ONCE-3DLanes has one semantic lane class.
num_category = 1
max_lanes = 20
class_names = ['lane']

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# Preserve the native cam01 aspect ratio at a 960-pixel input width.
org_h = 1020
org_w = 1920
resize_h = 510
resize_w = 960
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# Official ONCE evaluation covers x in [-10, 10] m and forward distance
# in [0, 50] m.  LaneLoom uses x=lateral, y=forward, z=height.
point_cloud_range = [-10.0, 0.0, -5.0, 10.0, 50.0, 5.0]
load_dim = 4
point_features = ['x', 'y', 'z', 'intensity']
use_dim = len(point_features)

# ONCE lane labels have no track IDs.  Keep the first integration single-frame
# rather than applying OpenLane's track-aware temporal-GT supplementation.
temporal_sampling_distance = False
temporal_sampling_enabled = False
smooth_3d_gt = True
gt_outlier_residual = 0.75

anchor_y_steps = [float(index) for index in range(1, 51)]

packed_keys = [
    'img',
    'lidar',
    'img_metas',
    'gt_lanes_3d',
    'gt_lanes_uv',
    'gt_lanes_visibility',
    'gt_lanes_category',
    'gt_lane_track_ids',
    'ground_lanes',
    'raw_lanes_3d',
    'raw_lanes_uv',
    'raw_lanes_uvy',
    'raw_label',
]

train_pipeline = [
    dict(
        type='LoadONCEImages',
        num_views=1,
        image_augmentation_mode='none',
        image_augmentation_recipes=[]),
    dict(type='ResizeONCEImages', keep_ratio=True),
    dict(type='NormalizeONCEImages', **img_norm_cfg),
    dict(type='LoadONCEPoints', load_dim=load_dim, use_dim=use_dim),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='LoadONCEAnnotations'),
    dict(type='PackInputs', keys=packed_keys),
]

test_pipeline = [
    dict(
        type='LoadONCEImages',
        num_views=1,
        image_augmentation_mode='none',
        image_augmentation_recipes=[]),
    dict(type='ResizeONCEImages', keep_ratio=True),
    dict(type='NormalizeONCEImages', **img_norm_cfg),
    dict(type='LoadONCEPoints', load_dim=load_dim, use_dim=use_dim),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadONCEAnnotations'),
    dict(type='PackInputs', keys=packed_keys),
]
eval_pipeline = test_pipeline

seed = 0
batch_size_per_gpu = 4
val_batch_size_per_gpu = 8
num_workers = 4
persistent_workers = False

dataset_cfg = dict(
    dataset_name=dataset_name,
    data_label_dir=os.path.join(once_root, 'ONCE_3DLanes'),
    data_image_dir=os.path.join(once_root, '3D_images', 'data'),
    lidar_processed_dir=os.path.join(once_root, '3D_lidars', 'data'),
    info_dir=os.path.join(once_root, '3D_infos', 'data'),
    split_dir=os.path.join(once_root, 'LaneLoom_splits'),
    # This machine currently has p0+p1 LiDAR only.  Use ``train`` here only
    # after all p0-p9 image/LiDAR/info files have been prepared.
    train_split='train_p0p1',
    val_split='val',
    test_split='test',
    org_h=org_h,
    org_w=org_w,
    resize_h=resize_h,
    resize_w=resize_w,
    anchor_y_steps=anchor_y_steps,
    train_pipeline=train_pipeline,
    val_pipeline=test_pipeline,
    num_category=num_category,
    max_lanes=max_lanes,
    class_names=class_names,
)
data_image_dir = dataset_cfg['data_image_dir']

train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='once_collate'),
    dataset=dict(
        type=dataset_type,
        split='train',
        temporal_sampling_distance=temporal_sampling_distance,
        smooth_3d_gt=smooth_3d_gt,
        gt_outlier_residual=gt_outlier_residual,
        dataset_cfg=dataset_cfg,
        modality=input_modality,
        test_mode=False),
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='once_collate'),
    dataset=dict(
        type=dataset_type,
        split='val',
        temporal_sampling_distance=temporal_sampling_distance,
        smooth_3d_gt=smooth_3d_gt,
        gt_outlier_residual=gt_outlier_residual,
        dataset_cfg=dataset_cfg,
        modality=input_modality,
        test_mode=True),
)

# ONCE test labels are private.  Local quantitative testing therefore uses
# the public validation set; official test submission export is a later flow.
test_dataloader = val_dataloader
