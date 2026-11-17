"""Direct geometric readout of predicted row evidence into five cubic templates."""
_base_ = ['./laneloom_openlane_q0_guided_spatial.py']

work_dir = './work_dirs/q0_bev_holdout/field_curve/supervised'
model = dict(lane_head=dict(anchor_prior=dict(geometric_readout=True,
                                              pc_range=[-25, 0, -5, 25, 100, 5])))
