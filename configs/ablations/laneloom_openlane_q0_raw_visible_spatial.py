"""Change only the extractor on the raw-visible, segment-held-out protocol."""
_base_ = ['./laneloom_openlane_q0_raw_visible.py']

work_dir = './work_dirs/q0_bev_holdout/raw_visible/spatial'
model = dict(lane_head=dict(anchor_prior=dict(
    type='SpatialBEVShapePrior', longitudinal_rows=50,
    spatial_layers=2, num_heads=4)))
