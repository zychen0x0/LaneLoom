"""Same geometric field head and losses; correct the input BEV coordinate frame."""
_base_ = ['./laneloom_openlane_q0_field_curve.py']

work_dir = './work_dirs/q0_bev_holdout/aligned/field_curve'
model = dict(pts_bev_encoder=dict(align_to_lane_ground=True))
