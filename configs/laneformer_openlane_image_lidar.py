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
        # LiDAR predicts one road curvature/grade pair.  It initializes 50
        # one-metre Bézier anchors and samples image features along them.
        anchor_prior=dict(
            type='LaneFormerLiDARBezierAnchorPrior',
            lidar_in_channels=512,
            image_in_channels=_dim_,
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            anchor_y_steps=[2.0 * i for i in range(51)],
            num_anchors=50,
            hidden_channels=64,
            # At y=100 m these bounds imply at most 25 m lateral bending and
            # 5 m height change, matching the decoder's configured xyz range.
            max_curvature=0.005,
            max_grade=0.05,
            curvature_loss_weight=1.0,
            grade_loss_weight=1.0),
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
            lidar_bev_adapter=dict(
                type='LaneFormerLiDARBEVAdapter',
                embed_dims=_dim_,
                lidar_in_channels=512),
            lidar_fusion=dict(
                type='LaneFormerImageLiDARFusion',
                embed_dims=_dim_,
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
            num_query=50,
            # OpenLane categories 0..20; category 21 is merged into 20 by dataset.
            num_category=21,
            anchor_y_steps=[2.0 * i for i in range(51)],
            # Layer 0 refines the LiDAR anchors, then the refined curves are
            # reprojected and sampled from image features for Layer 1.
            num_layers=2,
            num_heads=8,
            feedforward_channels=_ffn_dim_,
            dropout=0.1,
            visibility_temperature=0.75,
            # Median visible interval from the OpenLane training anchors.
            visibility_start_prior=0.14,
            visibility_end_prior=0.58,
            # Start with almost every anchor as background; Top2 assignment
            # then promotes only anchors close to a GT lane.
            objectness_prior_probability=0.01),
        loss_module=dict(
            type='LaneFormerTopKLaneLoss',
            # Each GT owns its two nearest initial Bézier anchors.  All other
            # anchors are background because there are only 50 candidates.
            assigner=dict(type='LaneFormerTopKAssigner', pos_k=2),
            cls_loss_weight=1.0,
            x_loss_weight=1.0,
            z_loss_weight=1.0,
            visibility_loss_weight=1.0,
            aux_loss_weight=0.5,
            smooth_l1_beta=0.5,
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
