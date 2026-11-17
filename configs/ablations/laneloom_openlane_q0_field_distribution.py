"""Keep the geometric head; align its position objective and probability readout.

Untrained candidate pending the ongoing one-epoch sigmoid control comparison.
The normalized multi-peak row target uses all annotated lanes, not one class
per template. No change to the 21-frame/74-GT evaluation or query lattice.
"""
_base_ = ['./laneloom_openlane_q0_field_curve.py']

work_dir = './work_dirs/q0_bev_holdout/field_distribution/supervised'
model = dict(lane_head=dict(anchor_prior=dict(lane_field_mode='row_distribution')))
