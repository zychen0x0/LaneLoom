"""Direct field-to-cubic geometry with auxiliary labels disabled."""
_base_ = ['./laneloom_openlane_q0_field_curve.py']

work_dir = './work_dirs/q0_bev_holdout/field_curve/no_aux'
q0_lane_field_loss_weight = 0.
