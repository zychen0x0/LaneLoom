"""D5 ablation for visible-point, axis-separated CQG supervision.

This keeps the existing five templates, 505 expanded queries and decoder.
The complete model is trained from scratch for 24 epochs. The decoder keeps
all existing detection losses, while Q0 is the CQG's only direct shape loss.
"""

_base_ = ['../main/laneloom_openlane_image_lidar_temporal_improve.py']

# Match configs/_base_/datasets/openlane_datasets_D5.py: train/validation use
# every fifth frame, while the final test loader remains full resolution.
dataset_name = 'openlane_D5'
load_interval = 5
dataset_cfg = dict(
    dataset_name=dataset_name,
    load_interval=load_interval)
test_dataset_cfg = dict(
    dataset_name=dataset_name,
    load_interval=1)
train_dataloader = dict(dataset=dict(dataset_cfg=dataset_cfg))
val_dataloader = dict(dataset=dict(dataset_cfg=dataset_cfg))
test_dataloader = dict(dataset=dict(dataset_cfg=test_dataset_cfg))

# Match the historical D5 run: two GPUs with per-GPU train/val batches 8/12.
train_dataloader['batch_size'] = 8
val_dataloader['batch_size'] = 12
test_dataloader['batch_size'] = 12

work_dir = './work_dirs/laneloom_openlane_image_lidar_temporal_improve_D5'
load_from = None

# Isolate the CQG supervision change. Keep the decoder's original visibility
# objective instead of mixing the separate balanced-BCE ablation into it.
model = dict(
    lane_head=dict(
        # Match the historical D5 decoder protocol. These are evidence
        # sampling locations only; output_y_steps still contains 50 points.
        lane_decoder=dict(
            evidence_y_steps=[float(y) for y in range(5, 101, 5)]),
        loss_module=dict(visibility_loss_type='smooth_l1')))

# Keep the original D5/24e optimizer so the historical epoch-17 result is a
# controlled comparison. The model/loss changes come from the improve base.
learning_rate = 2e-4

total_epochs = 24
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

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2,
        save_best='OpenLaneTemporal/F1',
        rule='greater'))
