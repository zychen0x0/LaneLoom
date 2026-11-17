"""Isolate native-resolution measured geometry as input to the same Q0 head."""
_base_ = ['./laneloom_openlane_q0_aligned_field_curve.py']

work_dir = './work_dirs/q0_bev_holdout/aligned/field_geometry'
model = dict(
    pts_bev_encoder=dict(return_road_evidence=True),
    lane_head=dict(anchor_prior=dict(use_road_evidence=True)))
