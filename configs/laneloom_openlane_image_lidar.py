# ---------------------------------------------------base config start---------------------------------------------------
_base_ = [
    './_base_/datasets/openlane_datasets_D5.py',
    './_base_/schedules/schedule_24e_adamw_cosine.py',
]
# ----------------------------------------------------base config end----------------------------------------------------

default_scope = 'mmdet3d'
launcher = 'none'
# tools/train.py creates a unique exp_date_YYYYMMDD_HHMMSS subdirectory here.
work_dir = './work_dirs/laneloom_openlane_image_lidar'
# Temporal history and LiDAR availability are data-dependent.  A rank can
# legitimately bypass a temporal or LiDAR projection on an individual step,
# so DDP must detect unused parameters instead of assuming a static graph.
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
)


# ---------------------------------------------------model common config start---------------------------------------------------
# Keep dataset filtering, Dynamic Pillar, query references and fusion range
# in the same metric space.
point_cloud_range = [-25.0, 0.0, -5.0, 25.0, 100.0, 5.0]
_dim_ = 256
_lidar_dim_ = 128
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
# The sparse backbone downsamples the 0.25 m pillar grid by four, then the
# lane neck upsamples once. The resulting LiDAR sampling field is 0.5 m.
bev_h_ = 200
bev_w_ = 100
queue_length = 3
output_y_steps = [2.0 * i for i in range(1, 51)]
evidence_y_steps = [5.0 * i for i in range(1, 21)]
score_threshold = 0.5

backbone_init = 'torchvision://resnet50'
# ----------------------------------------------------model common config end----------------------------------------------------


# ---------------------------------------------------model config start---------------------------------------------------
model = dict(
    type='LaneLoom',
    use_grid_mask=True,
    freeze_img=False,
    # Must match the dataset queue labels: oldest history, newest history,
    # current. History selection itself remains distance-first.
    frames=(-10, -5, 0),
    num_levels=_num_levels_,

    # ---------------------------------------------------image branch start---------------------------------------------------
    # Shared image encoder for t-10/t-5/t: one high-resolution P3 feature.
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=backbone_init)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    # ----------------------------------------------------image branch end----------------------------------------------------

    # ---------------------------------------------------lidar branch start---------------------------------------------------
    # Statistics-enhanced dynamic PFE followed by a PillarNeXt-style sparse
    # ResNet and single-scale dilated lane neck.
    pts_bev_encoder=dict(
        type='LaneLoomDynamicPillarBEVEncoder',
        point_cloud_range=point_cloud_range,
        pillar_size=[0.25, 0.25],
        out_channels=_lidar_dim_,
        # PillarNeXt-T style sparse widths: 32 -> 64 -> 128.
        base_channels=32,
        queue_length=queue_length),
    # ----------------------------------------------------lidar branch end----------------------------------------------------

    # ---------------------------------------------------lane head start---------------------------------------------------
    # LiDAR geometry prior plus curve-aligned temporal FV/LiDAR sampling.
    lane_head=dict(
        type='LaneLoomBEVLaneHead',
        # LiDAR predicts five translation-invariant road-shape tokens.
        anchor_prior=dict(
            type='LaneLoomLiDARShapeTokenPrior',
            lidar_in_channels=_lidar_dim_,
            pc_range=point_cloud_range,
            anchor_y_steps=output_y_steps,
            max_templates=5,
            hidden_channels=64,
            # Scales normalize the six unbounded slope outputs; they are not
            # hard limits and do not restrict cubic-Bézier expressiveness.
            lateral_slope_scale=0.1,
            vertical_slope_scale=0.03,
            yaw_scale_degrees=5.0,
            pitch_scale_degrees=1.0,
            curvature_scale=0.002,
            vertical_curvature_scale=0.001,
            shape_softmin_temperature=0.25,
            # Curves are the same shape only after optimal x/z translation
            # when P80 <= 0.3m, mean yaw <= 1deg and mean pitch <= 0.5deg.
            aligned_p80_threshold=0.3,
            yaw_similarity_degrees=1.0,
            pitch_similarity_degrees=0.5,
            min_lane_points=2,
            min_mode_points=4,
            short_support_length=20.0,
            shape_nms_threshold=1.0,
            shape_cover_loss_weight=1.0,
            # Five is a capacity ceiling. GT shape modes are matched one to
            # one, and only templates assigned to distinct modes are forced
            # apart; a straight-road sample is not forced to invent 5 modes.
            shape_mode_loss_weight=1.0,
            shape_diversity_loss_weight=0.05),
        # Include both -25m and +25m at an exact 0.5m step:
        # 5 templates x 101 seeds = 505 unfiltered Q0 queries.
        proposal_expander=dict(
            type='LaneLoomShapeProposalExpander',
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            anchor_y_steps=output_y_steps,
            num_x_seeds=101,
            template_feature_channels=64,
            q0_distance_threshold=0.3,
            q0_match_ratio=0.8,
            q0_geometry_loss_weight=1.0),
        bev_encoder=dict(
            type='LaneLoomBEVEncoder',
            embed_dims=_dim_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            lidar_bev_adapter=dict(
                type='LaneLoomLiDARBEVAdapter',
                embed_dims=_lidar_dim_,
                lidar_in_channels=_lidar_dim_)),
        query_sampler=dict(
            type='LaneLoomDualPlaneTemporalSampler',
            image_in_channels=_dim_,
            # Project sampled 128-D lane evidence to decoder width only after
            # curve-aligned sampling, instead of widening the complete map.
            lidar_in_channels=_lidar_dim_,
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            num_image_levels=_num_levels_,
            temporal_kernel_size=3,
            num_image_offsets=4,
            num_lidar_offsets=4,
            max_image_offset_pixels=2.0,
            max_lidar_offset_residual=0.5),
        lane_decoder=dict(
            type='LaneLoomLaneQueryDecoder',
            embed_dims=_dim_,
            bev_h=bev_h_,
            bev_w=bev_w_,
            pc_range=point_cloud_range,
            num_query=505,
            # OpenLane categories 0..20; category 21 is merged into 20 by dataset.
            num_category=21,
            anchor_y_steps=output_y_steps,
            evidence_y_steps=evidence_y_steps,
            # Each layer samples temporal FV and LiDAR-BEV along its input
            # curve. Layer 0 consumes Q0 and emits Q1; Layer 1 consumes Q1.
            num_layers=2,
            # Layer 0 sees all 505 Q0 curves. Layer 1 keeps 64 proposals:
            # 3.2x the dataset's max_lanes=20 and 4x a 16-lane scene.
            layer_query_budget=64,
            num_heads=8,
            feedforward_channels=_ffn_dim_,
            dropout=0.1,
            visibility_temperature=0.75,
            # Median visible interval from the OpenLane training anchors.
            visibility_start_prior=0.14,
            visibility_end_prior=0.58,
            # Start with almost every query as background; one-to-one matching
            # then promotes exactly one query for each visible GT lane.
            objectness_prior_probability=0.01),
        loss_module=dict(
            type='LaneLoomOneToOneLaneLoss',
            assigner=dict(type='LaneLoomHungarianAssigner'),
            objectness_loss_weight=1.0,
            category_loss_weight=0.25,
            # Positive/negative objectness are normalized separately, so an
            # equal group weight cannot be amplified by the 505-query count.
            # Near-GT unmatched candidates remain weaker ambiguous negatives.
            negative_objectness_weight=1.0,
            ambiguous_negative_weight=0.1,
            ambiguous_distance_threshold=0.5,
            ambiguous_match_ratio=0.75,
            diagnostic_geometry_distance_threshold=0.3,
            diagnostic_geometry_match_ratio=0.8,
            x_loss_weight=1.0,
            z_loss_weight=1.0,
            visibility_loss_weight=1.0,
            aux_loss_weight=0.5,
            smooth_l1_beta=0.5,
            focal_alpha=0.25,
            focal_gamma=2.0,
            diagnostic_score_threshold=score_threshold),
        test_cfg=dict(
            score_threshold=score_threshold,
            # Keep light geometric de-duplication as an inference safeguard.
            curve_nms=dict(
                visibility_threshold=0.5,
                distance_threshold=0.3,
                match_ratio_threshold=0.75))),
)
# ----------------------------------------------------model config end----------------------------------------------------

# MMEngine owns DDP, optimizer, scheduler, checkpointing, validation and test.
# The metric uses the pose-aligned temporal GT supplement, matching training.
# Official OpenLane case selection is performed by ``tools/test.py``.
default_hooks = dict(logger=dict(type='LaneLoomLoggerHook', interval=50))
val_evaluator = dict(
    type='OpenLaneAnchorMetric',
    prefix='OpenLane',
    score_threshold=score_threshold,
    visibility_threshold=0.5,
    distance_threshold=0.5,
    match_ratio_threshold=0.75,
    near_distance=40.0,
    use_temporal_gt_supplement=True,
    collect_device='cpu')
test_evaluator = val_evaluator
