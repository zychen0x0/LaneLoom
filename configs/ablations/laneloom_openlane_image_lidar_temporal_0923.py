"""D5 full-template CQG supervision with 0.1 x 0.2 m LiDAR pillars."""

_base_ = ['./laneloom_openlane_image_lidar_temporal_improve.py']

work_dir = './work_dirs/laneloom_openlane_image_lidar_temporal_0923_z_confidence_full_50e'
load_from = None
output_y_steps = [float(y) for y in range(5, 101, 5)]

train_dataset_cfg = dict(
    dataset_name='openlane_D5',
    load_interval=1,
    anchor_y_steps=output_y_steps,
    full_shape_template_targets=True,
    max_shape_templates=5,
    # Hard full-range extrapolation needs enough visible support to be stable.
    full_shape_min_points=4,
    full_shape_min_span=15.0,
    # A 4-point lane may borrow z evidence only from a compatible long lane.
    full_shape_guarded_short_merge=True,
    # Keep nearby extrapolation; progressively weaken unsupported far-end z.
    full_shape_z_extrapolation_grace_m=20.0,
    full_shape_z_extrapolation_decay_m=20.0)
val_dataset_cfg = dict(
    dataset_name='openlane_D5',
    load_interval=1,
    anchor_y_steps=output_y_steps,
    full_shape_template_targets=False)
test_dataset_cfg = dict(
    dataset_name='openlane_D5',
    load_interval=1,
    anchor_y_steps=output_y_steps,
    full_shape_template_targets=False)

train_dataloader = dict(
    batch_size=8,
    dataset=dict(dataset_cfg=train_dataset_cfg))
val_dataloader = dict(dataset=dict(dataset_cfg=val_dataset_cfg))
test_dataloader = dict(dataset=dict(dataset_cfg=test_dataset_cfg))

model = dict(
    pts_bev_encoder=dict(pillar_size=[0.1, 0.2]),
    lane_head=dict(
        use_enhanced_prior_gt=False,
        anchor_prior=dict(
            anchor_y_steps=output_y_steps,
            supervise_shapes=True,
            shape_supervision='full_templates'),
        proposal_expander=dict(anchor_y_steps=output_y_steps),
        # Use the encoder's 250 x 250 BEV for both shape prior and decoder.
        bev_encoder=dict(bev_h=250, bev_w=250),
        lane_decoder=dict(
            anchor_y_steps=output_y_steps,
            evidence_y_steps=output_y_steps),
        loss_module=dict(use_enhanced_gt=False)))

# One optimizer update per microbatch: single-GPU batch size is exactly eight.
optim_wrapper = dict(accumulative_counts=1)

learning_rate = 2e-4
total_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=total_epochs,
    val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=learning_rate * 1e-3,
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs),
]

# Keep validation/checkpoint selection aligned with the decoder's official
# current-frame GT rather than the temporal GT supplement inherited upstream.
val_evaluator = dict(
    prefix='OpenLaneOfficial',
    use_temporal_gt_supplement=False)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        save_best='OpenLaneOfficial/F1',
        rule='greater'))
