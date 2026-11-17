"""Replace only the BEV-to-template extractor on the common holdout split."""
_base_ = ['./laneloom_openlane_q0_holdout.py']

work_dir = './work_dirs/laneloom_openlane_q0_spatial'

# Preserve five global cubic templates, 101 seeds and decoder/loss settings.
# Height and RGB information come from the same 128-channel BEV as baseline.
model = dict(lane_head=dict(anchor_prior=dict(
    type='SpatialBEVShapePrior', longitudinal_rows=50,
    spatial_layers=2, num_heads=4)))
