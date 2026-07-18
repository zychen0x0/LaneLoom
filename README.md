# LaneFormer


## LiDAR Shape-Token Q0

当前 LiDAR 几何先验不再回归单一 `curvature/grade`。训练流程为：

```text
LiDAR-BEV → 10 Shape Tokens → shape clustering (K=1..10)
          → K×50 x seeds → current-frame P3 Proposal Router Top-50
          → two-layer curve decoder
```

模板只学习相对三维形状，监督由
`loss_lidar_shape_cover + loss_lidar_shape_compact` 构成，不使用模板
confidence、类别或可见区间。`loss_proposal_router` 使用当前图像证据监督模板
的绝对横向放置。Decoder 输入仍固定为 50 条 Q0。

每个 Shape Token 只预测 6 个无界几何量：三段横向斜率和三段纵向
斜率。`0.1/0.03` 仅作为输出数值尺度，不使用 `tanh/clamp`，因此不限制
三次 Bézier 的形状表达。三段斜率被累积转换为四个控制点；固定的
`x_seed` 只在 Router 阶段做严格横向平移。控制多边形三段的
`yaw/pitch` 角度由斜率反算，仅用于诊断；Shape Coverage 的方向与曲率
仍从采样后的完整曲线计算。

由于 LiDAR prior 参数结构已经改变，旧版单一 `curvature/grade` checkpoint
不能完整恢复新模块；新实验应从头训练，或仅选择性加载兼容的 backbone 权重。


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
work_dirs/laneformer_openlane_image_lidar/exp_date_20260715_191319/checkpoints/best_OpenLane_F1_epoch_22.pth \
--work-dir work_dirs/laneformer_openlane_image_lidar/nms_evaluation \
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

cd /workspace/LaneFormer

tensorboard \
  --logdir work_dirs/laneformer_openlane_image_lidar/exp_date_20260715_191319/Tensorboard \
  --host 0.0.0.0 \
  --port 6006
