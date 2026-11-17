"""Observed road evidence and continuous paths on the target-segment holdout."""
_base_ = ['./laneloom_openlane_q0_holdout.py']

work_dir = './work_dirs/q0_bev_holdout/continuous'
train_dataset_cfg = dict(full_shape_visible_fit_tolerance_m=.01)
train_dataloader = dict(dataset=dict(dataset_cfg=train_dataset_cfg))
model = dict(
    pts_bev_encoder=dict(return_road_evidence=True),
    lane_head=dict(anchor_prior=dict(
        type='ContinuousBEVShapePrior', pc_range=[-25, 0, -5, 25, 100, 5],
        longitudinal_rows=50, max_step_m=4., motion_scale_m=.75,
        # Match the baseline objectives for the architecture comparison.
        # Auxiliary direction/curvature losses belong to a separate ablation.
        direction_loss_weight=0., curvature_loss_weight=0.)))
