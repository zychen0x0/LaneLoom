"""E2c (optional): masked temporal mean instead of TConv (else = E0)."""

_base_ = ['./e0_full_50e.py']

work_dir = './work_dirs/ablation_e2c_mean_instead_of_tconv_50e'

model = dict(
    lane_head=dict(query_sampler=dict(type='AblationMeanTimeSampler')))
