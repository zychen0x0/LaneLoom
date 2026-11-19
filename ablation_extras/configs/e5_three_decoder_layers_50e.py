"""E5 (optional): three decoder layers instead of two (else = E0)."""

_base_ = ['./e0_full_50e.py']

work_dir = './work_dirs/ablation_e5_three_decoder_layers_50e'

model = dict(lane_head=dict(lane_decoder=dict(num_layers=3)))
