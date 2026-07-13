# ---------------------------------------------------base config start---------------------------------------------------
_base_ = [
    './_base_/datasets/openlane_datasets_D5.py',
    './_base_/schedules/schedule_24e_adamw_cosine.py',
]
# ----------------------------------------------------base config end----------------------------------------------------

default_scope = 'mmdet3d'
launcher = 'none'
# tools/train.py creates a unique exp_date_YYYYMMDD_HHMMSS subdirectory here.
work_dir = './work_dirs/laneformer_openlane_image_lidar'
# Temporal history and LiDAR availability are data-dependent.  A rank can
# legitimately bypass a temporal or LiDAR projection on an individual step,
# so DDP must detect unused parameters instead of assuming a static graph.
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)


# ---------------------------------------------------model common config start---------------------------------------------------
# Front-view OpenLane BEV range. Keep dataset point filtering, LiDAR voxel grid,
# BEV query reference points and fusion pc_range in the same metric space.
point_cloud_range = [-25.0, 0.0, -5.0, 25.0, 100.0, 5.0]
voxel_size = [0.125, 0.125, 0.25]
out_size_factor = 8

_dim_ = 256
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 50
queue_length = 3
num_points_in_pillar = 5
reference_zs = [-5.0, -2.0, 0.0, 2.0, 5.0]

backbone_init = 'torchvision://resnet50'
# ----------------------------------------------------model common config end----------------------------------------------------


# ---------------------------------------------------model config start---------------------------------------------------
model = dict(
    type='LaneFormer',
    use_grid_mask=True,
    freeze_img=False,
    # Must match the dataset queue labels: oldest history, newest history,
    # current. History selection itself remains distance-first.
    frames=(-10, -5, 0),
    num_levels=_num_levels_,

    # ---------------------------------------------------image branch start---------------------------------------------------
    # Image encoder: ResNet50 + single-level FPN(C5).
    # Keep it simple and stable before replacing it with CBSwinTransformer + FPNC.
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=backbone_init)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    # ----------------------------------------------------image branch end----------------------------------------------------

    # ---------------------------------------------------lidar branch start---------------------------------------------------
    # LiDAR branch. The dataset keeps queue_index as the
    # last point channel; LaneFormer strips it before voxel encoding.
    pts_input_dim=5,
    pts_voxel_layer=dict(
        max_num_points=5,
        voxel_size=voxel_size,
        max_voxels=150000,
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 800, 400],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # ----------------------------------------------------lidar branch end----------------------------------------------------

    # ---------------------------------------------------lane head start---------------------------------------------------
    # Per-frame image/LiDAR BEV fusion, pose-aligned temporal fusion and lane
    # set prediction with independent LaneFormer loss weights.
    lane_head=dict(
        type='LaneFormerBEVLaneHead',
        bev_encoder=dict(
            type='LaneFormerBEVEncoder',
            embed_dims=_dim_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            spatial_cross_attn=dict(
                type='LaneFormerSpatialCrossAttention',
                embed_dims=_dim_,
                bev_h=bev_h_,
                bev_w=bev_w_,
                pc_range=point_cloud_range,
                num_points_in_pillar=num_points_in_pillar,
                reference_zs=reference_zs,
                num_levels=_num_levels_),
            lidar_fusion=dict(
                type='LaneFormerImageLiDARFusion',
                embed_dims=_dim_,
                lidar_in_channels=512,
                fusion_method='cross_attention',
                num_layers=2,
                num_heads=8,
                feedforward_channels=_ffn_dim_,
                dropout=0.1),
            temporal_self_attn=dict(
                type='LaneFormerTemporalSelfAttention',
                embed_dims=_dim_,
                bev_h=bev_h_,
                bev_w=bev_w_,
                pc_range=point_cloud_range)),
        lane_decoder=dict(
            type='LaneFormerLaneQueryDecoder',
            embed_dims=_dim_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            num_query=40,
            # OpenLane categories 0..20; category 21 is merged into 20 by dataset.
            num_category=21,
            anchor_y_steps=[2.0 * i for i in range(51)],
            num_layers=3,
            num_heads=8,
            feedforward_channels=_ffn_dim_,
            dropout=0.1,
            # Training-only DN-DETR lane queries reconstruct a GT Bézier
            # lane from noisy control points; inference remains 40 queries.
            dn_num_groups=1,
            dn_noise_scale=0.5,
            visibility_temperature=0.75,
            # Median visible interval from the OpenLane training anchors.
            visibility_start_prior=0.14,
            visibility_end_prior=0.58,
            # Start with almost every query as background; positives are
            # learned through assignment instead of emitting all 40 queries.
            objectness_prior_probability=0.01),
        loss_module=dict(
            type='LaneFormerHungarianLaneLoss',
            # Hierarchical supervision: all queries are screened by
            # objectness; lane attributes apply only after matching.
            geometry_loss_weight=10.0,
            visibility_loss_weight=1.0,
            visibility_dice_loss_weight=1.0,
            visibility_boundary_loss_weight=0.25,
            # Focal loss is normalized by positive lanes, so this no longer
            # needs to compensate for averaging over all 40 queries.
            objectness_loss_weight=2.0,
            category_loss_weight=0.25,
            dn_loss_weight=1.0,
            aux_loss_weight=0.5,
            geometry_huber_beta=0.5,
            # Keep metre-based geometry/Huber semantics while preventing the
            # 50 m x-range from dominating global gradient clipping.
            geometry_normalization=20.0,
            match_cost_geometry=1.0,
            match_cost_vis=0.5,
            match_cost_vis_dice=1.0,
            match_cost_objectness=1.0,
            memory_loss_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0),
        test_cfg=dict(
            score_threshold=0.3)),
)
# ----------------------------------------------------model config end----------------------------------------------------

# MMEngine owns DDP, optimizer, scheduler, checkpointing, validation and test.
# The metric uses the pose-aligned temporal GT supplement, matching training.
# Official OpenLane case selection is performed by ``tools/test.py``.
default_hooks = dict(logger=dict(type='LaneFormerLoggerHook', interval=50))
val_evaluator = dict(
    type='OpenLaneAnchorMetric',
    prefix='OpenLane',
    score_threshold=0.3,
    visibility_threshold=0.5,
    distance_threshold=0.5,
    match_ratio_threshold=0.75,
    near_distance=40.0,
    use_temporal_gt_supplement=True,
    collect_device='cpu')
test_evaluator = val_evaluator
