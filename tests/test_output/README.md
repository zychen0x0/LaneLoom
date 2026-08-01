# 可视化输出

'''bash

cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0 \
python tests/test_output/infer_openlane_sample.py \
  --image data/openlane/images/training/segment-3195159706851203049_2763_790_2783_790_with_camera_labels/150990953886019500.jpg \
  --checkpoint work_dirs/laneloom_openlane_image_lidar/exp_date_20260729_161015/checkpoints/best_OpenLaneTemporal_F1_epoch_23.pth \
  --output-dir tests/test_output/vis_output/fuza

'''



复杂车道线
data/openlane/images/training/segment-3195159706851203049_2763_790_2783_790_with_camera_labels/150990953886019500.jpg


遮挡
data/openlane/images/training/segment-8603916601243187272_540_000_560_000_with_camera_labels/155250133667392400.jpg


分叉
data/openlane/images/training/segment-17750787536486427868_560_000_580_000_with_camera_labels/155321858754444500.jpg


S型车道线
data/openlane/images/training/segment-16372013171456210875_5631_040_5651_040_with_camera_labels/150715623041978000.jpg


夜间低可见度
data/openlane/images/training/segment-9295161125729168140_1270_000_1290_000_with_camera_labels/151547628889630300.jpg


路口车道线断裂
data/openlane/images/training/segment-15903544160717261009_3961_870_3981_870_with_camera_labels/150732932818699600.jpg


U型车道线
data/openlane/images/training/segment-18418533015310829002_480_000_500_000_with_camera_labels/155008811255146600.jpg


车道线汇聚
data/openlane/images/training/segment-17750787536486427868_560_000_580_000_with_camera_labels/155321859534404000.jpg