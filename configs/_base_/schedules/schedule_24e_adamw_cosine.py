# ---------------------------------------------------optimizer wrapper start---------------------------------------------------
# LaneLoom uses an OpenLane-friendly AdamW setup with an MMEngine optimizer
# wrapper and cosine schedule.
learning_rate = 2e-4

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=learning_rate,
        betas=(0.95, 0.99),
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'image_offset_head': dict(lr_mult=0.1, decay_mult=1.0),
            'lidar_offset_head': dict(lr_mult=0.1, decay_mult=1.0),
        }),
    clip_grad=dict(max_norm=20, norm_type=2))
# ----------------------------------------------------optimizer wrapper end----------------------------------------------------


# ---------------------------------------------------schedule start---------------------------------------------------
total_epochs = 24
val_interval = 1
keep_n_ckpts = 2

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
        by_epoch=True,
        begin=0,
        end=total_epochs),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=total_epochs,
    val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='ExperimentCheckpointHook',
        interval=1,
        max_keep_ckpts=keep_n_ckpts,
        # Dataset/evaluator configs select the explicit benchmark metric.
        save_best=None,
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
load_from = None
# ----------------------------------------------------schedule end----------------------------------------------------
