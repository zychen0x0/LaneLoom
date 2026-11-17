"""Isolate head learning on a shared mature encoder; validation is evaluation-only."""
_base_ = ['./laneloom_openlane_q0_fulltrain.py']

work_dir = './work_dirs/q0_official_validation/frozen_encoder/baseline'
q0_freeze_encoder = True
q0_encoder_checkpoint = ('work_dirs/laneloom_openlane_image_lidar_temporal_0923_z_confidence_full_50e/'
                         'exp_date_20260924_174958/checkpoints/best_OpenLaneOfficial_F1_epoch_43.pth')
