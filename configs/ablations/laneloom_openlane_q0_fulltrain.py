"""All official training scenes are eligible; official validation stays unseen."""
_base_ = ['./laneloom_openlane_q0_aligned.py']

work_dir = './work_dirs/q0_official_validation/baseline'
q0_evaluation_split = 'validation'
q0_validation_scenes = 'docs/场景.md'
q0_excluded_segments = []
q0_target_frames = dict(_delete_=True)
train_dataset_cfg = dict(excluded_segments=[])
train_dataloader = dict(dataset=dict(dataset_cfg=train_dataset_cfg))
