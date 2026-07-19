# LiDAR-BEV 车道 Proposal 重设计调研

## 结论

用户提出的总体方向是可行的：

```text
点云 → 轻量二维 BEV 编码 → 5 个平移不变形状模板
     → 横向密集平移 → 约 500 条 Q0
     → 曲线局部双视图采样 → 两层 Decoder → 一对一匹配
```

但有四点需要先纠正：

1. **LiLaDet 并没有放弃 pillar/voxel。**它的 BEV Pathway 先 pillarize 点云，Spatial Pathway 又使用 3D sparse voxels；因此只能借鉴“粗 proposal + 局部三维 refinement”，不能把它当成无 voxel 点云骨干的依据。[LiLaDet 论文 §IV](https://arxiv.org/html/2309.13596v3#S4)
2. 将点云“打平”为 BEV 仍然需要离散的 \(x,y\) 网格，本质上属于 pillar/raster 表示，只是没有划分 \(z\) voxel。不能丢弃 \(z\)，应把高度统计保留为 BEV 通道。
3. 5 个模板在模板阶段不应与原始多条 GT 做一对一 Hungarian。多条平行车道可能具有同一种形状；若一对一匹配，会浪费模板。模板阶段应使用**平移不变的 many-to-one coverage**，最终 500 条实例 Q0 再使用一对一 Hungarian。
4. `Top1` 是训练样本分配，不会减少 500 条曲线的前向计算。固定让 500 条曲线进入 Decoder 后，应依赖 `objectness + Hungarian` 学习背景；推理阶段再用分数阈值和轻量 Curve NMS 去重。

## 1. LiLaDet 实际如何处理点云

LiLaDet 是双路径结构：

- BEV Pathway：raw points → pillarized pseudo-BEV → 多尺度 ViT → lane segmentation + height；
- Spatial Pathway：raw points → 3D voxelization → 多尺度 sparse 3D convolution；
- 对 BEV proposal point 使用 k-NN 搜索局部 voxel features，再通过 BVAT cross-attention 融合 BEV 与 spatial features并回归坐标偏移。[论文方法](https://arxiv.org/html/2309.13596v3#S4)

论文给出的具体设置是：

- pseudo-BEV \(x,y\) 分辨率：0.04 m；
- 3D voxel size：`[0.1, 0.1, 0.2]` m；
- 每 voxel 最多32点；
- 最多12000个 voxels；
- 每个 proposal 搜索12个邻近 voxel features；
- proposal 真值距离阈值：0.5 m。[论文实现细节](https://arxiv.org/html/2309.13596v3#S5.SS2)

所以 LiLaDet 的核心启发不是“直接压平即可”，而是：

> BEV 适合发现车道候选，但粗 BEV 会损失局部三维几何，需要在候选附近重新取得精细点云证据。

论文也明确指出 coarse BEV 无法准确恢复 proposal 的三维位置，因此引入 Spatial Pathway；这与当前项目计划的“沿 Q0 在 LiDAR-BEV 和图像 FV 做局部 deformable sampling”在功能上相近。[LiLaDet BEV/Spatial Pathway](https://arxiv.org/html/2309.13596v3#S4.SS2)

需要注意：LiLaDet 官方仓库目前只有 README，没有公开训练网络代码，无法直接复用其 backbone 实现。[LiLaDet 官方仓库](https://github.com/RunkaiZhao/LiLaDet)

## 2. 无 3D voxel 的轻量 LiDAR-BEV 是否可行

可行，但准确说法应是：

```text
取消 z 方向体素划分
保留 x-y pillar/raster 网格
在每个网格中聚合点特征
```

K-Lane/RLLDN 的官方活跃配置提供了一个很接近的参考实现：

- 点云直接投影成 \(1152\times1152\) 的三通道 BEV image；
- 三个通道为 `z / intensity / reflectivity`；
- 使用 ResNet34 提取二维特征；
- 再用 ViT Global Feature Correlator 和 row-wise head 检测车道。[官方配置](https://github.com/kaist-avelab/K-Lane/blob/main/configs/Proj28_GFC-T3_RowRef_82_73.py) [投影代码](https://github.com/kaist-avelab/K-Lane/blob/main/baseline/utils/pc_utils.py)

RLLDN 的第二阶段并非重新处理全图，而是围绕第一阶段每一行预测的 lane column，截取左右 `off_grid=2` 的局部窗口，再把现有 lane tokens 输入 Transformer 做 lane correlation refinement。[官方 refinement head](https://github.com/kaist-avelab/K-Lane/blob/main/baseline/models/heads/row_shared_not_reduc_ref.py)

这支持当前项目使用：

```text
粗 Q0 决定采样中心
    ↓
在曲线左右采样局部特征
    ↓
进行局部 refinement
```

而不是直接做昂贵的全图 image cross-attention。

但 K-Lane 的简单 rasterizer 在同一格出现多个点时会覆盖已有值，并不显式保留点数与高度分布。当前项目更推荐每个 \(x,y\) cell 聚合：

```text
log(point_count)
z_min / z_max / z_mean / z_std
intensity_max / intensity_mean
reflectivity_max / reflectivity_mean（若数据提供）
RGB或图像语义特征（可选）
```

这样仍然是轻量二维 BEV，不需要 3D sparse voxel，同时不会把道路坡度所需的 \(z\) 信息直接丢掉。

另一个边界是：K-Lane 官方配置只覆盖前方约46.08 m，不能用它的结果直接证明这种表示在 OpenLane 的100 m范围仍然可靠。[官方 ROI 配置](https://github.com/kaist-avelab/K-Lane/blob/main/configs/Proj28_GFC-T3_RowRef_82_73.py)

## 3. 5 个 Shape Templates 应如何监督

### 3.1 平移不变的形状表示

对 GT 在合法 `y_steps` 上取点，以第一个合法点为参考：

\[
\tilde{x}(y)=x(y)-x(y_{\mathrm{ref}}),\qquad
\tilde{z}(y)=z(y)-z(y_{\mathrm{ref}})
\]

预测模板也做相同归一化。模板覆盖距离可以写成：

\[
d_{\mathrm{shape}}(T_k,G_j)=
\frac{1}{|S_j|}
\sum_{y\in S_j}
\left\|\tilde{T}_k(y)-\tilde{G}_j(y)\right\|_1
+\lambda_{\mathrm{dir}}
\left\|\Delta\tilde{T}_k(y)-\Delta\tilde{G}_j(y)\right\|_1
\]

其中 \(S_j\) 只用于 **loss 中的 GT support mask**。推理 forward 不得读取 GT support、GT visibility 或 GT 模板数量。

仅比较坐标差容易让局部方向不准，因此建议同时监督一阶切向差。这里不监督绝对横向位置，符合“模板只学习形状”的目标。

### 3.2 不应使用模板到原始 GT 的一对一匹配

假设一张图有四条近似平行车道，它们的绝对位置不同，但平移归一化后可能属于同一种形状。模板阶段正确的 coverage 是：

\[
L_{\mathrm{cover}}=
\frac{1}{N_{\mathrm{GT}}}
\sum_j \operatorname{SoftMin}_{k=1..5}
d_{\mathrm{shape}}(T_k,G_j)
\]

这允许一个模板覆盖多条形状相同的 GT。再加入轻量 diversity/usage 约束，避免5个模板完全重合；但不要强迫每张图必须使用5种形状。

模板输出与最终实例匹配应明确分层：

```text
5个模板：many-to-one shape coverage，只学形状
        ↓
横向平移形成约500条Q0
        ↓
Decoder最终预测：Hungarian one-to-one，只学实例归属
```

LiLaDet 本身也观察到真实车道曲率多样，固定线形 anchors 若要覆盖全部位置和形状会需要大量预定义 anchors；这正好支持“少量学习形状 × 密集位置平移”，但论文没有验证这里提出的5模板具体数量。[LiLaDet 数据分析](https://arxiv.org/html/2309.13596v3#S3.SS3)

### 3.3 Q0 质量应有两级标准

模板的 shape-only loss 不可能保证平移后的 Q0 达到绝对评测精度。因此需要同时存在：

1. `loss_lidar_shape_cover`：忽略平移，只判断形状；
2. `loss_q0_geometry`：对平移后的候选与 GT 做绝对逐点覆盖，标准严格于最终评测，例如 `0.3m / 80%`。

否则可能出现“曲线形状相似，但所有 x seeds 或高度基准均没有形成合格 proposal”的情况。

LiDAR 实际可观测范围约70 m时，70–100 m 分支只能由数据先验预测，不能保证 sample-specific 正确。GT support 可以用于训练 loss mask，这不属于数据泄漏；但 GT support、GT visibility 和 GT lane count 不能进入 Q0 forward。远端应主要由图像 FV sampling 和 Decoder 修复。

## 4. 约500条 Q0 与显存

`[-25,25]` 若包含两端并按0.5 m采样，实际有：

\[
(25-(-25))/0.5+1=101
\]

所以5个模板会生成 **505条**，不是500条。若严格需要500条，可使用100个0.5 m bin centers：

```text
-24.75, -24.25, ..., 24.75
```

0–100 m 每5 m采样且包含两端时，每条曲线有21个采样点。

曲线局部采样的基础位置数为：

```text
505 queries × 21 points × (3个历史FV + 1个LiDAR-BEV)
= 42,420 locations / sample / decoder layer
```

这通常比全局 image cross-attention 更可控。以256维 FP16 特征计算，仅保存每个基础位置的采样特征约20.7 MiB/sample/layer；若每点使用4个 deformable offsets，尚未聚合的激活会再放大约4倍，训练时还需考虑梯度与两层 Decoder。

因此推荐：

- 图像只使用 ResNet50 的一个 P3 或 P4 尺度；
- 每条曲线使用21个5 m采样点；
- 每个点先使用2–4个法向局部 offsets；
- 对不同历史帧的采样结果尽早做时序聚合，不长期保留全部高维张量；
- query self-attention 可以保留，避免对全部 FV tokens 做全局 cross-attention。

`Top1` 或 Hungarian 不会减少上述采样量；若500条全部进入 Decoder，这是固定计算成本。

## 5. PointPainting：可以做，但不建议只拼原始 RGB

原始 PointPainting 并不是给点云追加 RGB，而是：

```text
图像语义分割网络
    ↓
逐像素类别分数
    ↓
点投影到图像
    ↓
把对应像素的类别分数追加到每个点
```

其目的在于给点云添加紧凑的语义信息，且可接入 PointPillars、VoxelNet、PointRCNN 等不同点云网络。[PointPainting 原论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Vora_PointPainting_Sequential_Fusion_for_3D_Object_Detection_CVPR_2020_paper.pdf)

直接追加 `R/G/B` 可以作为低成本实验，但存在：

- 光照、阴影、曝光和天气变化；
- 标定或时间不同步时采错颜色；
- 遮挡关系导致后方点采到前景物体颜色；
- 当前图像分支已经提供视觉特征，重复输入可能弱化 LiDAR-only 几何先验的定义。

更推荐的优先级是：

```text
第一版：z + intensity + density等纯LiDAR统计
第二版：追加当前图像backbone的浅层特征或lane/road语义logits
对照组：追加原始RGB
```

使用输入图像预测得到的特征不属于 GT 泄漏；但必须保证 painting 的 forward 不读取验证 GT、GT lane mask 或 GT support。若加入图像特征，这个分支应称为“painted LiDAR-BEV proposal”，不能再作为严格的 LiDAR-only 分支做消融。

## 6. 推荐的最终实现边界

```text
历史点云姿态对齐并拼接
        ↓
Point-wise features
[x,y,z,intensity]
        ↓
动态x-y scatter聚合，无z voxel
        ↓
单尺度2D BEV backbone
        ↓
5个相对三次Bezier Shape Templates
        ↓ shape-only many-to-one coverage
5 × 101 = 505条平移Q0
        ↓
按5m间隔在3帧FV和LiDAR-BEV做曲线局部deformable sampling
        ↓
Temporal fusion + query self/cross update
        ↓
两层geometry/visibility/objectness/category refinement
        ↓
训练：Hungarian one-to-one
推理：score threshold + 轻量Curve NMS
```

这个方案比直接复制 LiLaDet 更符合当前 LaneFormer，但建议先做三个逐步消融：

1. 纯 LiDAR 2D scatter + 5 templates；
2. 增加505 Q0 双视图局部采样；
3. 未来可单独评估 PointPainting 特征。

否则同时替换点云编码、Q0生成、采样方式和图像语义增强后，即使 F1 改变，也无法判断收益来自哪一项。

> 当前实现和本轮实验明确不加入 PointPainting；这里只将其作为未来独立消融项。
