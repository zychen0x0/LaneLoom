"""Matched D5/24-epoch control using LaneLoom's dynamic pillars."""

_base_ = ['../main/laneloom_openlane_image_lidar.py']

work_dir = './work_dirs/laneloom_openlane_image_lidar_dp'

# One OpenLane sample in five for training/validation; keep final test full.
load_interval = 5
dataset_name = 'openlane_D5'
dataset_cfg = dict(dataset_name=dataset_name, load_interval=load_interval)
test_dataset_cfg = dict(dataset_name=dataset_name, load_interval=1)

# Keep the optimizer batch identical across all three encoder ablations.
batch_size_per_gpu = 4
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    dataset=dict(dataset_cfg=dataset_cfg))
val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    dataset=dict(dataset_cfg=dataset_cfg))
test_dataloader = dict(
    batch_size=batch_size_per_gpu,
    dataset=dict(dataset_cfg=test_dataset_cfg))

total_epochs = 24
learning_rate = 2e-4
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
    val_interval=1)

# 25 / 256 m gives an exact 1024 x 512 XY grid. It is within 2.4% of
# LaneLoom's 0.1 m baseline and remains divisible through official backbones.
xy_cell_size = 0.09765625
model = dict(
    pts_bev_encoder=dict(pillar_size=[xy_cell_size, xy_cell_size]))
