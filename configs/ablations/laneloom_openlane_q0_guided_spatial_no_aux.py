"""Same guided architecture, without auxiliary field labels: loss ablation."""
_base_ = ['./laneloom_openlane_q0_guided_spatial.py']

work_dir = './work_dirs/q0_bev_holdout/guided_spatial/no_aux'
q0_lane_field_loss_weight = 0.
