"""Same frozen encoder/data/budget, native-grid geometry-guided template head."""
_base_ = ['./laneloom_openlane_q0_frozen_encoder.py']

work_dir = './work_dirs/q0_official_validation/frozen_encoder/field_geometry'
q0_lane_field_loss_weight = 1.
model = dict(
    pts_bev_encoder=dict(return_road_evidence=True),
    lane_head=dict(anchor_prior=dict(
        type='GuidedSpatialBEVShapePrior', geometric_readout=True,
        pc_range=[-25, 0, -5, 25, 100, 5], use_road_evidence=True)))
