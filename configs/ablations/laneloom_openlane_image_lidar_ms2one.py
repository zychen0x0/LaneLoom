_base_ = ['../main/laneloom_openlane_image_lidar_Temporal.py']

# Keep ablation outputs separate from the checkpoint-compatible main run.
work_dir = './work_dirs/laneloom_openlane_image_lidar_ms2one'

model = dict(
    img_backbone=dict(
        out_indices=(1, 2, 3)),
    img_neck=dict(
        in_channels=[512, 1024, 2048],
        num_outs=3),
    # DV-3DLane-style multi-scale-to-one fusion. The decoder still samples
    # one 1/8-resolution FV map; C4/C5 context is merged into that map here.
    img_ms2one=dict(
        type='DilatedMultiScaleFusion',
        in_channels=256,
        out_channels=256,
        num_scales=3,
        dilations=(1, 2, 5)),
)
