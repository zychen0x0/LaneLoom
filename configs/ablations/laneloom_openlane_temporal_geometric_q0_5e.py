"""Five-epoch end-to-end fine-tuning with the validated geometric Q0 prior.

Initialize the full detector with tools/prepare_q0_finetune_checkpoint.py:
epoch-43 image/decoder weights plus the matched step-21000 LiDAR/Q0 pair.
The optimizer and five-epoch learning-rate schedule start anew.
"""
_base_ = ['../main/laneloom_openlane_image_lidar_temporal.py']

work_dir = './work_dirs/laneloom_openlane_temporal_geometric_q0_5e'
load_from = work_dir + '/initialization.pth'
resume = False

model = dict(
    pts_bev_encoder=dict(pillar_size=[0.1, 0.2], align_to_lane_ground=True),
    lane_head=dict(
        q0_raw_visible_supervision=True,
        q0_lane_field_loss_weight=1.0,
        anchor_prior=dict(
            type='GuidedSpatialBEVShapePrior',
            supervise_shapes=False,
            longitudinal_rows=50, spatial_layers=2, num_heads=4,
            geometric_readout=True,
            pc_range=[-25, 0, -5, 25, 100, 5])))

# Two GPUs x four samples = effective batch eight, without accumulation.
train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
                        prefetch_factor=1,
                        dataset=dict(dataset_cfg=dict(scene_cache_size=16)))
val_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
                      prefetch_factor=1,
                      dataset=dict(dataset_cfg=dict(scene_cache_size=16)))
test_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
                       prefetch_factor=1,
                       dataset=dict(dataset_cfg=dict(scene_cache_size=16)))

# This is fine-tuning pretrained modules, rather than training from scratch.
# Reduce updates on the validated LiDAR/Q0 pair while adapting the decoder.
learning_rate = 2e-5
optim_wrapper = dict(
    accumulative_counts=1,
    optimizer=dict(lr=learning_rate),
    paramwise_cfg=dict(custom_keys={
        'pts_bev_encoder': dict(lr_mult=0.25),
        'lane_head.anchor_prior': dict(lr_mult=0.25)}))
total_epochs = 5
train_cfg = dict(max_epochs=total_epochs, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=1/3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=learning_rate*1e-3,
         T_max=total_epochs, by_epoch=True, begin=0, end=total_epochs)]

# Train on the full official training split. Evaluate final detections against
# current-frame official GT, independently of temporal GT supplementation.
val_evaluator = dict(prefix='OpenLaneOfficial', use_temporal_gt_supplement=False)
test_evaluator = dict(prefix='OpenLaneOfficial', use_temporal_gt_supplement=False)
default_hooks = dict(
    checkpoint=dict(save_best='OpenLaneOfficial/F1', max_keep_ckpts=5, interval=1),
    logger=dict(brief_metrics=['loss_score', 'loss_category', 'loss_x', 'loss_z',
                               'loss_visibility', 'loss_tube', 'loss_q0_raw_geometry',
                               'loss_bev_lane_position', 'loss_bev_lane_height']))
randomness = dict(seed=7)
