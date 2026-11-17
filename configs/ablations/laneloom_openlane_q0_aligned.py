"""Original head with point BEV coordinates calibrated to the unchanged lane GT."""
_base_ = ['./laneloom_openlane_q0_raw_visible.py']

work_dir = './work_dirs/q0_bev_holdout/aligned/baseline'
model = dict(pts_bev_encoder=dict(align_to_lane_ground=True))
