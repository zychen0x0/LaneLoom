"""Coverage diagnostic: supervise Q0 from observed lanes, without full extrapolation.

Opt-in experiment; the epoch 43 baseline and its checkpoint stay unchanged.
This tests a training-target change, not an inference threshold adjustment.
"""

_base_ = ['./laneloom_openlane_image_lidar_temporal_0923.py']

work_dir = './work_dirs/laneloom_openlane_q0_visible_coverage'

# Without synthetic full-lane fields, BEVLaneHead.loss uses current-frame GT
# coordinates and visibility for the existing Q0 geometry loss.
train_dataset_cfg = dict(full_shape_template_targets=False)
train_dataloader = dict(dataset=dict(dataset_cfg=train_dataset_cfg))

# Mode matching and coverage use observed lane support. Do not keep the
# full-template loss when disabling the full-template target generator.
model = dict(lane_head=dict(
    anchor_prior=dict(shape_supervision='visible_modes')))
