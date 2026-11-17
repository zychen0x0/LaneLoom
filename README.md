# LaneLoom

## 配置目录

- `configs/main/`：已验证的主实验配置；旧 checkpoint 应使用这里的配置。
- `configs/ablations/`：消融实验配置；每项消融使用独立文件和输出目录。


## 多卡训练

```bash
cd /workspace/LaneLoom

# 单帧模型训练
PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/ablations/laneloom_openlane_image_lidar_temporal_0923.py \
--launcher pytorch


PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/main/laneloom_once_image_lidar_temporal.py \
--launcher pytorch \
--load-from "work_dirs/laneloom_once_image_lidar/exp_date_20261010_111319/checkpoints/best_ONCEOfficial_F1_epoch_2.pth"
```

## 继续训练 load_from

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/main/laneloom_openlane_image_lidar_temporal.py \
--launcher pytorch \
--load-from "work_dirs/laneloom_openlane_image_lidar/exp_date_20260803_081030/checkpoints/best_OpenLaneTemporal_F1_epoch_44.pth"

```

## 评估 checkpoint

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29502 \
tools/test.py \
configs/main/laneloom_openlane_image_lidar_Temporal.py \
work_dirs/laneloom_openlane_image_lidar/exp_date_20260803_081030/checkpoints/best_OpenLaneTemporal_F1_epoch_44.pth \
--case validation \
--launcher pytorch
```

默认输出位置
--work-dir work_dirs/laneloom_openlane_image_lidar/exp_date_20260803_081030 \

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
  --logdir work_dirs/laneloom_once_image_lidar/exp_date_20261010_165137/Tensorboard \
  --host 0.0.0.0 \
  --port 6006
