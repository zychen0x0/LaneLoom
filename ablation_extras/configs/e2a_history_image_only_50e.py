"""E2a: historical images in MTFF, current LiDAR sweep only (else = E0)."""

_base_ = ['./e0_full_50e.py']

work_dir = './work_dirs/ablation_e2a_history_image_only_50e'

model = dict(pts_bev_encoder=dict(type='AblationCurrentSweepPillarEncoder'))
