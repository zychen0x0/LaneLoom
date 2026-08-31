"""Single-frame OpenLane image + LiDAR training and official evaluation."""

_base_ = ['./laneloom_openlane_image_lidar_Temporal.py']

# Keep the paper's single-frame baseline separate from temporal experiments.
work_dir = './work_dirs/laneloom_openlane_image_lidar_single_frame'

# The parent config has already resolved its temporal interpolation, so disable
# history explicitly at every data/model consumer below.
temporal_sampling_distance = False
validation_metric_prefix = 'OpenLaneOfficial'
validation_best_metric = 'OpenLaneOfficial/F1'

# The temporal main config inherits the D5 development recipe. Paper results use
# every OpenLane training, validation, and test annotation instead.
dataset_name = 'openlane'
load_interval = 1
dataset_cfg = dict(dataset_name=dataset_name, load_interval=load_interval)
test_dataset_cfg = dict(dataset_cfg)

train_dataloader = dict(
    dataset=dict(
        temporal_sampling_distance=temporal_sampling_distance,
        dataset_cfg=dataset_cfg))
val_dataloader = dict(
    dataset=dict(
        temporal_sampling_distance=temporal_sampling_distance,
        dataset_cfg=dataset_cfg))
test_dataloader = dict(
    dataset=dict(
        temporal_sampling_distance=temporal_sampling_distance,
        dataset_cfg=test_dataset_cfg))

model = dict(
    pts_bev_encoder=dict(
        temporal_sampling_distance=temporal_sampling_distance))

val_evaluator = dict(
    prefix=validation_metric_prefix,
    use_temporal_gt_supplement=False)

default_hooks = dict(
    checkpoint=dict(save_best=validation_best_metric))
