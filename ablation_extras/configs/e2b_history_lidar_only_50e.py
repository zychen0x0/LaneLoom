"""E2b: accumulated LiDAR sweeps, current image only in MTFF (else = E0)."""

_base_ = ['./e0_full_50e.py']

work_dir = './work_dirs/ablation_e2b_history_lidar_only_50e'

model = dict(
    lane_head=dict(query_sampler=dict(type='AblationCurrentImageSampler')))
