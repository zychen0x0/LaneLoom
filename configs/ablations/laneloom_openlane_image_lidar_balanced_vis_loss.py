"""Single-frame OpenLane ablation with balanced visibility supervision."""

_base_ = ['../main/laneloom_openlane_image_lidar.py']

work_dir = './work_dirs/laneloom_openlane_image_lidar_balanced_vis_loss'

model = dict(
    lane_head=dict(
        loss_module=dict(
            visibility_loss_type='balanced_bce',
            visibility_loss_weight=1.0,
            visibility_start_loss_weight=0.5,
            visibility_end_loss_weight=0.5)))
