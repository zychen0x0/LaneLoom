# LaneFormer


## LiDAR Shape-Token Q0

当前 LiDAR 几何先验不再回归单一 `curvature/grade`。训练流程为：

```text
点云 → 保留高度/密度统计的二维 Dynamic Pillar → LiDAR-BEV
      → 5 Shape Tokens
      → 每个模板平移到 [-25,25] 的101个0.5m横向种子
      → 5×101=505条 Q0（不做 Router 预筛）
      → P3 FV + LiDAR-BEV 双平面局部 deformable sampling
      → two-layer curve decoder
```

模板只学习相对三维形状，不使用模板 confidence 或类别。训练期只把 GT
可见性作为 loss 支撑区间，不输入 LiDAR Shape Prior，因此不会泄漏给推理。
两条曲线先在共同 y 支撑上做最优 x/z 平移对齐；默认同时满足
`P80 <= 0.3m`、平均 `yaw <= 1°`、平均 `pitch <= 0.5°` 才视为同一形状。

Shape Prior 使用三项监督：

- `loss_lidar_shape_cover`：每条 GT 至少有一条形状相近的模板，短 GT 按
  y 支撑长度降低权重；
- `loss_lidar_shape_mode`：先用 complete-link 将 GT 聚成形状模态，再用
  Hungarian 将各模态一对一匹配到不同模板；只有至少含4个共同合法锚点的
  GT 才定义可靠的全局形状模态，2～3点短片段只参与逐 lane coverage；
- `loss_lidar_shape_diversity`：只分离已经负责不同 GT 模态的模板。5 是容量
  上限，不会在只有一种直行形状的样本中强迫五个模板达到代数秩5。

日志中的 `diag_gt_shape_metric_rank` 与
`diag_predicted_shape_metric_rank` 是 GT 和预测模板在上述阈值下的“度量
覆盖秩”，不是 5×6 参数矩阵的普通矩阵秩。`diag_shape_lane_cover_recall`
和 `diag_shape_mode_cover_recall` 分别检查逐车道与逐形状模态的覆盖率。

当前 5 个模板全部参与 Q0 展开。`loss_q0_geometry` 直接约束送入 Decoder 的505条 Q0，
要求至少 80% 的 GT 可见点误差小于 0.3m，比最终评测的
0.5m/75% 标准更严格。Decoder 输入固定为505条 Q0，并通过 Hungarian
一对一匹配保证每条 GT 只监督一个 query。

每个 Shape Token 只预测 6 个无界几何量：三段横向斜率和三段纵向
斜率。`0.1/0.03` 仅作为输出数值尺度，不使用 `tanh/clamp`，因此不限制
三次 Bézier 的形状表达。三段斜率被累积转换为四个控制点；固定的
`x_seed` 只在 proposal expansion 阶段做严格横向平移。控制多边形三段的
`yaw/pitch` 角度由斜率反算，仅用于诊断；Shape Coverage 的方向与曲率
仍从采样后的完整曲线计算。

由于 LiDAR prior 参数结构已经改变，旧版单一 `curvature/grade` checkpoint
不能完整恢复新模块；新实验应从头训练，或仅选择性加载兼容的 backbone 权重。

当前实现和本轮实验不加入 PointPainting；它仅保留为未来单独消融项，
避免把点云编码、Q0生成、双视图采样和图像语义增强的收益混在一起。


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
