"""Calibrated input, with matching row-distribution supervision and readout."""
_base_ = ['./laneloom_openlane_q0_field_distribution.py']

work_dir = './work_dirs/q0_bev_holdout/aligned/field_distribution'
model = dict(pts_bev_encoder=dict(align_to_lane_ground=True))
