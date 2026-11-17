"""Predicted lane-field guidance; target segments remain entirely held out."""
_base_ = ['./laneloom_openlane_q0_raw_visible_spatial.py']

work_dir = './work_dirs/q0_bev_holdout/guided_spatial/supervised'
q0_lane_field_loss_weight = 1.
model = dict(lane_head=dict(anchor_prior=dict(type='GuidedSpatialBEVShapePrior')))
