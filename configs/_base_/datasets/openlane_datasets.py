import os


# ---------------------------------------------------dataset settings start---------------------------------------------------
dataset_type = 'OpenLaneDataset'
dataset = '1000'
dataset_name = 'openlane'

laneloom_root = os.path.abspath(os.path.join('{{ fileDirname }}', '..', '..', '..'))
local_data_root = os.path.join(laneloom_root, 'data')
data_root = os.environ.get('LANELOOM_DATA_ROOT', local_data_root)

# OpenLane lane category ids. Category 21 is merged into category 20.
num_category = 21
max_lanes = 20
class_names = [
    'invalid',
    'white-dash',
    'white-solid',
    'double-white-dash',
    'double-white-solid',
    'white-ldash-rsolid',
    'white-lsolid-rdash',
    'yellow-dash',
    'yellow-solid',
    'double-yellow-dash',
    'double-yellow-solid',
    'yellow-ldash-rsolid',
    'yellow-lsolid-rdash',
    'fishbone',
    'others',
    'lane-category-15',
    'lane-category-16',
    'lane-category-17',
    'lane-category-18',
    'lane-category-19',
    'roadedge',
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
# ----------------------------------------------------dataset settings end----------------------------------------------------


# ---------------------------------------------------image settings start---------------------------------------------------
# Image preprocessing is declared explicitly in train_pipeline/test_pipeline.
# img_scale follows MMDetection's (short_edge, long_edge) convention.
org_h = 1280
org_w = 1920
img_scale = (640, 960)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Experiment ladder for image + LiDAR 3D lane detection:
#   original          原始 image + LiDAR, num_views can be 1 or more
#   random_aug        每个 view 都使用同一套随机颜色/压缩/噪声增强策略
#   multi_aug_views   num_views=6 时使用原图 + 5 个固定场景增强视图
# TODO: view dropout 训练时随机丢增强视图
# TODO: attention fusion 用 attention 替代 mean fusion

num_views = 1
view_mode = 'original'

# 6 个 view 的顺序是
# original     原图
# brightness   明暗/过曝/夜间鲁棒
# blur         细雨/轻微模糊
# blur         高斯模糊，模拟雨水/镜头污染
# occlusion    遮挡，模拟车流遮挡车道线
# random       随机颜色/压缩/噪声

multi_aug_view_transforms = [
    dict(name='original', transforms=[]),
    dict(
        name='brightness',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=(-0.45, 0.35),
                contrast_limit=(-0.25, 0.25),
                p=1.0),
        ]),
    dict(
        name='fog',
        transforms=[
            dict(
                type='RandomFog',
                fog_coef_lower=0.24,
                fog_coef_upper=0.42,
                alpha_coef=0.10,
                p=1.0),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=(-0.08, 0.04),
                contrast_limit=(-0.18, -0.04),
                p=1.0),
        ]),
    dict(
        name='blur',
        transforms=[
            dict(
                type='RandomRain',
                slant_lower=-8,
                slant_upper=8,
                drop_length=18,
                drop_width=1,
                drop_color=(170, 170, 170),
                blur_value=5,
                brightness_coefficient=0.90,
                rain_type='drizzle',
                p=1.0),
            dict(
                type='GaussianBlur',
                blur_limit=(3, 7),
                sigma_limit=(0.4, 1.4),
                p=1.0),
        ]),
    dict(
        name='occlusion',
        transforms=[
            dict(
                type='CoarseDropout',
                max_holes=18,
                max_height=96,
                max_width=96,
                min_holes=8,
                min_height=32,
                min_width=32,
                fill_value=0,
                p=1.0),
        ]),
    dict(
        name='random',
        transforms=[
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='ColorTint',
                        color=(255, 174, 64),
                        alpha=0.30,
                        p=1.0),
                    dict(
                        type='ColorTint',
                        color=(255, 35, 18),
                        alpha=0.24,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=(-8, 14),
                        sat_shift_limit=(15, 40),
                        val_shift_limit=(-35, 10),
                        p=1.0),
                ],
                p=1.0),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='ImageCompression', quality_lower=65, quality_upper=90, p=1.0),
                    dict(type='GaussNoise', var_limit=(15.0, 70.0), p=1.0),
                    dict(type='RandomToneCurve', scale=0.18, always_apply=None, p=1.0),
                ],
                p=0.65),
        ]),
]
# ----------------------------------------------------image settings end----------------------------------------------------


# ---------------------------------------------------bev/lidar settings start---------------------------------------------------
# Front-view OpenLane lane detection only uses the vehicle-forward region.
# x: left/right [-50m, 50m], y: forward [0m, 100m], z: height [-5m, 5m].
point_cloud_range = [-25.0, 0.0, -5.0, 25.0, 100.0, 5.0]
load_dim = 6
use_dim = 6
# ----------------------------------------------------bev/lidar settings end----------------------------------------------------


# ------------------------------------------------temporal sampling start-------------------------------------------------
# Temporal queue labels: oldest history, newest history, current.
# Selection remains distance-first (5m / 10m); these labels match the explicit
# fallback offsets used when those distance targets are unavailable.
frames = (-10, -5, 0)
history_num_frames = len([frame for frame in frames if frame < 0])

# If both distance values are positive, OpenLaneDataset samples history at
# approximately 5m and 10m behind the current ego pose.
history_distance_m = 10.0
history_distance_step_m = 5.0
# A distance slot that cannot be satisfied falls back to these *frame offsets*
# (oldest to newest): t-10 and t-5.  Every modality is still loaded using the
# selected fallback frame key; no current-frame metadata is reused.
history_fallback_frame_offsets = (10, 5)
# Preserve the current-frame GT tensors and additionally emit a separately
# gated temporal supplement target for temporal-GT experiments.
temporal_gt_augmentation = True
# Fit every full lane with one robust cubic Bezier x(y), z(y) curve before
# fixed-y anchoring. Raw JSON geometry remains available for auditing.
smooth_3d_gt = True
gt_outlier_residual = 0.75
# -------------------------------------------------temporal sampling end--------------------------------------------------


# ---------------------------------------------------lane gt settings start---------------------------------------------------
# Fifty uniformly spaced forward anchors at 2 m intervals. The first output
# anchor is y=2 m; y=0 is excluded while the 100 m boundary is retained.
anchor_y_steps = [2.0 * i for i in range(1, 51)]
num_y_steps = len(anchor_y_steps)
# ----------------------------------------------------lane gt settings end----------------------------------------------------


# ---------------------------------------------------augmentation start---------------------------------------------------
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.),
        ],
        p=0.5),
    dict(
        type='PixelDropout',
        dropout_prob=0.01,
        per_channel=False,
        drop_value=0,
        mask_drop_value=None,
        always_apply=None,
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=1),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1),
            dict(type='RandomToneCurve', scale=0.1, always_apply=None, p=1.0),
        ],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=1.0),
            dict(
                type='ZoomBlur',
                max_factor=(1, 1.05),
                step_factor=(0.01, 0.01),
                always_apply=None,
                p=1.0),
        ],
        p=0.3),
]

image_augmentation = dict(
    albumentations=dict(
        type='Albu',
        transforms=albu_train_transforms),
    grid_mask=dict(
        type='GridMask',
        ratio=0.5,
        prob=0.7,
        # mode=0 preserves LaneLoom's existing mask. mode=1 selects its
        # complement, matching the mode used by BEVFormer's default config.
        mode=0),
)

packed_keys = [
    'img',
    'lidar',
    'img_metas',
    'gt_lanes_3d',
    'gt_lanes_uv',
    'gt_lanes_visibility',
    'gt_lanes_category',
    'gt_lane_track_ids',
    'gt_lanes_3d_enhanced',
    'gt_lanes_visibility_enhanced',
    'gt_lanes_category_enhanced',
    'gt_lane_track_ids_enhanced',
    'ground_lanes',
    'raw_lanes_3d',
    'raw_lanes_uv',
    'raw_lanes_uvy',
    'raw_label',
]

train_pipeline = [
    dict(
        type='LoadTemporalOpenLaneImages',
        num_views=num_views,
        view_mode=view_mode,
        multi_aug_view_transforms=multi_aug_view_transforms),
    dict(type='ResizeOpenLaneImages', img_scale=img_scale, keep_ratio=True),
    image_augmentation['albumentations'],
    dict(type='NormalizeOpenLaneImages', **img_norm_cfg),
    image_augmentation['grid_mask'],
    dict(
        type='LoadPoseAlignedOpenLanePoints',
        load_dim=load_dim,
        use_dim=use_dim),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='LoadOpenLaneAnnotations'),
    dict(type='PackInputs', keys=packed_keys),
]

test_pipeline = [
    dict(
        type='LoadTemporalOpenLaneImages',
        num_views=num_views,
        view_mode=view_mode,
        multi_aug_view_transforms=multi_aug_view_transforms),
    dict(type='ResizeOpenLaneImages', img_scale=img_scale, keep_ratio=True),
    dict(type='NormalizeOpenLaneImages', **img_norm_cfg),
    dict(
        type='LoadPoseAlignedOpenLanePoints',
        load_dim=load_dim,
        use_dim=use_dim),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='LoadOpenLaneAnnotations'),
    dict(type='PackInputs', keys=packed_keys),
]
eval_pipeline = test_pipeline
# ----------------------------------------------------augmentation end----------------------------------------------------


# ---------------------------------------------------data config start---------------------------------------------------
seed = 0
batch_size_per_gpu = 8
val_batch_size_per_gpu = 12
num_workers = 4
persistent_workers = False

dataset_cfg = dict(
    dataset=dataset,
    dataset_name=dataset_name,
    data_label_dir=os.path.join(data_root, 'openlane', 'lane3d_1000'),
    data_image_dir=os.path.join(data_root, 'openlane', 'images'),
    lidar_processed_dir=os.path.join(data_root, 'waymo', 'waymo_processed_data'),
    train_split='training',
    val_split='validation',
    test_split='test',
    org_h=org_h,
    org_w=org_w,
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
    collate_fn=dict(type='openlane_collate'),
    dataset=dict(
        type=dataset_type,
        split='train',
        frames=frames,
        history_num_frames=history_num_frames,
        history_distance_m=history_distance_m,
        history_distance_step_m=history_distance_step_m,
        history_fallback_frame_offsets=history_fallback_frame_offsets,
        temporal_gt_augmentation=temporal_gt_augmentation,
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
    collate_fn=dict(type='openlane_collate'),
    dataset=dict(
        type=dataset_type,
        split='val',
        frames=frames,
        history_num_frames=history_num_frames,
        history_distance_m=history_distance_m,
        history_distance_step_m=history_distance_step_m,
        history_fallback_frame_offsets=history_fallback_frame_offsets,
        temporal_gt_augmentation=temporal_gt_augmentation,
        smooth_3d_gt=smooth_3d_gt,
        gt_outlier_residual=gt_outlier_residual,
        dataset_cfg=dataset_cfg,
        modality=input_modality,
        test_mode=True),
)

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='openlane_collate'),
    dataset=dict(
        type=dataset_type,
        # Do not reuse validation here: lane3d_1000/test has its own labels.
        split='test',
        frames=frames,
        history_num_frames=history_num_frames,
        history_distance_m=history_distance_m,
        history_distance_step_m=history_distance_step_m,
        history_fallback_frame_offsets=history_fallback_frame_offsets,
        temporal_gt_augmentation=temporal_gt_augmentation,
        smooth_3d_gt=smooth_3d_gt,
        gt_outlier_residual=gt_outlier_residual,
        dataset_cfg=dataset_cfg,
        modality=input_modality,
        test_mode=True),
)
# ----------------------------------------------------data config end----------------------------------------------------
