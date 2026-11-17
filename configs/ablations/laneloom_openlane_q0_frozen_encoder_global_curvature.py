"""Global learned curvature supports the measured local curve where LiDAR ends."""
_base_ = ['./laneloom_openlane_q0_frozen_encoder_field_geometry.py']

work_dir = './work_dirs/q0_official_validation/frozen_encoder/global_curvature'
model = dict(lane_head=dict(anchor_prior=dict(
    curvature_prior_strength=1e-4, curvature_prior_scale=20.)))
