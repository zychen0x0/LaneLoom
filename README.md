# LaneLoom


## 多卡训练

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/laneloom_openlane_image_lidar.py \
--launcher pytorch

```

## 继续训练 load_from

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/laneloom_openlane_image_lidar.py \
--launcher pytorch \
--load-from "work_dirs/laneloom_openlane_image_lidar/exp_date_20260725_163203/checkpoints/epoch_24.pth"

```

## 评估 checkpoint

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29502 \
tools/test.py \
configs/laneloom_openlane_image_lidar.py \
work_dirs/laneloom_openlane_image_lidar/exp_date_20260715_191319/checkpoints/best_OpenLane_F1_epoch_22.pth \
--work-dir work_dirs/laneloom_openlane_image_lidar/nms_evaluation \
--case validation \
--launcher pytorch
```


# 其他模式

# 验证集 + 全部测试场景（默认行为）
--case all

# 只跑验证集
--case validation

# 只跑测试集
--case test

# 只跑单个场景
--case curve_case
--case intersection_case
--case merge_split_case






## TensorBoard 可视化

cd /workspace/LaneLoom

tensorboard \
  --logdir work_dirs/laneloom_openlane_image_lidar/exp_date_20260720_220252/Tensorboard \
  --host 0.0.0.0 \
  --port 6006
