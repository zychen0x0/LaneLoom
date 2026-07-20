# LaneLoom


## LiDAR Shape-Token Q0

当前 LiDAR 几何先验不再回归单一 `curvature/grade`。训练流程为：

```text
点云 → 保留高度/密度统计的二维 Dynamic Pillar → LiDAR-BEV
      → 5 Shape Tokens
      → 每个模板平移到 [-25,25] 的101个0.5m横向种子
      → 5×101=505条 Q0（不做 Router 预筛）
      → P3 FV + LiDAR-BEV 双平面局部 deformable sampling
      → Decoder Layer 0处理505条
      → 根据Layer 0 objectness保留64条
      → Decoder Layer 1精修64条
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
覆盖秩”，不是 5×6 参数矩阵的普通矩阵秩。
`diag_shape_mode_cover_recall` 检查不同 GT 形状模态是否都被模板覆盖。
训练日志只保留这三个 Shape Prior 诊断量，避免与三项实际 loss 重复。

当前 5 个模板全部参与 Q0 展开。`loss_q0_geometry` 直接约束送入 Decoder 的505条 Q0，
要求至少 80% 的 GT 可见点误差小于 0.3m，比最终评测的
0.5m/75% 标准更严格。Decoder Layer 0输入固定为505条 Q0，并通过
Hungarian一对一匹配保证每条 GT 只监督一个 query。

Layer 0仍使用全部505条 Q0，避免在获得图像证据前误删远端分叉。Layer 1
只保留64条：训练时强制包含Q0与Layer 0 Hungarian正样本的并集，推理时
完全使用Layer 0预测排序；`diag_layer0_topk_gt_recall` 只统计未强制的
原始Top-64是否以0.3m/80%标准覆盖GT，防止teacher forcing让诊断虚高。

分类监督分为 binary objectness 和仅正样本 category。正负objectness分别
归一化，整个负样本组权重为0.1；与GT在75%可见点上距离小于0.5m、但未被
Hungarian选中的相邻候选视为模糊负样本，再降权到0.1。这样保留一对一
训练目标，同时避免约500个背景query压制几何与前景学习。
推理筛选和Curve NMS只使用binary objectness；positive-only category
只决定已检测车道的类别，不会再次压低车道存在概率。

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
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29501 \
tools/train.py \
configs/laneloom_openlane_image_lidar.py \
--launcher pytorch
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

一次推理扫描多个 objectness 阈值：

```bash
PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29503 \
tools/test.py \
configs/laneloom_openlane_image_lidar.py \
path/to/checkpoint.pth \
--work-dir work_dirs/laneloom_openlane_image_lidar/threshold_scan \
--case validation \
--score-thresholds 0.3 0.5 0.6 0.7 0.8 \
--launcher pytorch
```

从已有权重启动新优化器，并只训练一轮做诊断：

```bash
PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29504 \
tools/train.py \
configs/laneloom_openlane_image_lidar.py \
--load-from path/to/checkpoint.pth \
--max-epochs 1 \
--work-dir work_dirs/laneloom_openlane_image_lidar/objectness_warmstart \
--launcher pytorch
```

`--load-from`只加载模型权重，不恢复旧优化器；不要与`--resume`同时使用。


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
  --logdir work_dirs/laneloom_openlane_image_lidar/exp_date_20260715_191319/Tensorboard \
  --host 0.0.0.0 \
  --port 6006
