"""Matched baseline: original Q0 head, original visible points, one geometry loss."""
_base_ = ['./laneloom_openlane_q0_holdout.py']

work_dir = './work_dirs/q0_bev_holdout/raw_visible/baseline'
q0_raw_visible_supervision = True
train_dataset_cfg = dict(full_shape_template_targets=False)
train_dataloader = dict(dataset=dict(dataset_cfg=train_dataset_cfg))
model = dict(lane_head=dict(anchor_prior=dict(supervise_shapes=False)))
