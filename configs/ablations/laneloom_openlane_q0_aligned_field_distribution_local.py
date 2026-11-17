"""Calibrated input and parallel, density-referenced row queries."""
_base_ = ['./laneloom_openlane_q0_field_distribution_local.py']

work_dir = './work_dirs/q0_bev_holdout/aligned/field_distribution_local'
model = dict(pts_bev_encoder=dict(align_to_lane_ground=True))
