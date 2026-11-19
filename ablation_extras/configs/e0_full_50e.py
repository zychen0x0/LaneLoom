"""E0: full LaneLoom (temporal) retrained under the shared ablation protocol.

Every ablation config inherits this file, so all rows of the ablation table
share one schedule: 50 epochs, two GPUs x 8 samples, 500-iteration linear
warm-up followed by cosine annealing, and a fixed random seed.

Validation inside training runs every 10 epochs (the base config validates
every epoch on the full split, which costs a large share of training time).
The run script evaluates the final-epoch checkpoint, so no checkpoint is
selected on the evaluation split.
"""

_base_ = ['../../configs/main/laneloom_openlane_image_lidar_temporal.py']

custom_imports = dict(
    imports=['LaneLoom.ablation_extras.modules'],
    allow_failed_imports=False)

work_dir = './work_dirs/ablation_e0_full_50e'

randomness = dict(seed=2026)

train_dataloader = dict(batch_size=8)

total_epochs = 50
train_cfg = dict(max_epochs=total_epochs, val_interval=10)
default_hooks = dict(checkpoint=dict(save_best=None))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=2e-7,
        T_max=total_epochs,
        by_epoch=True,
        begin=0,
        end=total_epochs),
]
