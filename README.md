# LaneFormer


## 多卡训练

```bash
cd /workspace/LaneFormer

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/laneformer_openlane_image_lidar.py \
--launcher pytorch
```


## 评估 checkpoint

```bash
cd /workspace/LaneFormer

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29502 \
tools/test.py \
configs/laneformer_openlane_image_lidar.py \
work_dirs/laneformer_openlane_image_lidar/exp_date_YYYYMMDD_HHMMSS/checkpoints/epoch_24.pth \
--launcher pytorch
```


## TensorBoard 可视化

cd /workspace/LaneFormer

tensorboard \
  --logdir work_dirs/laneformer_openlane_image_lidar/exp_date_20260712_192936/Tensorboard \
  --host 0.0.0.0 \
  --port 6006