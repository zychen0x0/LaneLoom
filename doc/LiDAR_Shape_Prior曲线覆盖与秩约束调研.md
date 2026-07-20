# LiDAR Shape Prior：曲线覆盖、有效模板数与“秩”约束调研

## 结论

当前问题不应该定义成：

> 让 5 组 Bézier 参数矩阵的代数秩等于 5。

更准确的定义是：

> 在“忽略绝对位置、只比较曲线形状”的距离空间里，用不超过 5 个模板覆盖当前样本的所有 GT 形状模式，并避免两个有效模板重复覆盖同一个模式。

因此，正确的数学量是 **固定误差阈值下的覆盖数（\(\varepsilon\)-covering number）或形状模态数**，不是原始 6 参数矩阵的 rank。

固定输出 5 个 Shape Token 时，5 表示最大容量；对单个样本真正应该有效的模板数为：

\[
M_{\mathrm{GT}}\leq 5
\]

不应对每个样本强制 \(M_{\mathrm{GT}}=5\)。如果当前道路只有一种近似直行形状，强制 5 个模板线性无关只会让另外 4 个模板生成没有 GT 依据的极端曲线。

推荐损失由三部分组成：

\[
L_{\mathrm{shape}}
=
\lambda_{\mathrm{lane}}L_{\mathrm{lane-cover}}
+
\lambda_{\mathrm{mode}}L_{\mathrm{mode-match}}
+
\lambda_{\mathrm{div}}L_{\mathrm{matched-diversity}}
\]

- `lane-cover`：每条合法 GT 至少能被一个模板近似；
- `mode-match`：不同 GT 形状模态一对一匹配不同模板；
- `matched-diversity`：仅约束已经匹配到不同 GT 模态的模板不要坍缩。

普通 rank、effective rank 和 log-det 可以作为诊断或弱正则，但不能替代基于曲线距离的覆盖损失。

---

## 1. 为什么原始 Bézier 参数矩阵的 rank 不等于表达能力

设 5 个模板各自预测 6 个斜率参数：

\[
T=
\begin{bmatrix}
t_1^\top\\
t_2^\top\\
\cdots\\
t_5^\top
\end{bmatrix}
\in\mathbb{R}^{5\times6}
\]

直接计算 \(\operatorname{rank}(T)\) 存在四个根本问题。

### 1.1 当前模型是“选择一个模板”，不是“线性组合模板”

稀疏字典学习中的 rank 和字典原子有明确含义：信号通过多个字典原子的稀疏线性组合重构。K-SVD 的原始定义就是使用过完备字典，并以稀疏线性组合表达信号。[Aharon、Elad、Bruckstein，K-SVD](https://doi.org/10.1109/TSP.2006.881199)

如果 LaneLoom 使用：

\[
g_j \approx \sum_k \alpha_{jk}t_k
\]

并显式学习稀疏系数 \(\alpha_{jk}\)，那么字典 rank、稀疏度和重构误差才属于同一套数学模型。

但当前 Shape Prior 的逻辑是：

\[
g_j \approx t_{k^*}
\]

即一条 GT 由一条模板加横向平移表达，不会把模板 1、模板 2 相加生成新车道。因此它更接近 k-medoids、vector quantization 或 metric covering，而不是线性字典分解。

### 1.2 线性相关不代表形状可以互相替代

一个最直接的反例是：

```text
straight = 0
left     = v
right    = -v
```

这三组参数的矩阵 rank 只有 1，但在曲线距离意义下：

- 直行模板不能代替左转；
- 左转模板不能代替右转；
- 三种模板都可能是当前场景必须保留的模式。

因此“rank 小”不能推出“模板表达能力不足”。

### 1.3 原始控制点 rank 会受到坐标原点和平移影响

如果使用绝对控制点而不是斜率：

- 同一形状仅改变 `x_seed`，参数矩阵 rank 可能上升；
- 对全部控制点增加相同坐标偏移，奇异值分布也会改变；
- 但这些曲线在本项目定义中仍是同一个形状模板。

所以在没有先消除平移的情况下，控制点参数 rank 甚至不是一个形状不变量。

### 1.4 rank 衡量线性子空间维数，不衡量阈值覆盖数

一组形状可能全部位于一个二维非线性流形上，线性 rank 很低，却需要很多模板才能在 0.3 m 误差内覆盖。反过来，少量带噪声的形状可能在数值上满秩，却只需要一个模板。

Roy 和 Vetterli 提出的 effective rank 用归一化奇异值的 Shannon entropy 给出连续的有效维数：

\[
p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
\operatorname{erank}(T)=
\exp\left(-\sum_i p_i\log p_i\right)
\]

它解决了普通 rank 对微小噪声非常敏感的问题，但仍然描述的是**有效线性维数**，不是需要多少个最近邻模板才能覆盖 GT。[Roy、Vetterli，Effective Rank](https://infoscience.epfl.ch/entities/publication/26352894-e3af-4922-aea4-8aab418b0358)

因此：

- effective rank 可用于观察模板参数是否数值坍缩；
- 不能把 `effective_rank == GT 模态数` 当作主训练目标。

---

## 2. 正确概念：形状空间中的最小覆盖数

设当前样本的合法 GT 形状集合为：

\[
\mathcal{G}=\{G_1,\ldots,G_J\}
\]

模型预测模板：

\[
\mathcal{T}=\{T_1,\ldots,T_5\}
\]

定义平移不变的曲线形状距离 \(d_{\mathrm{shape}}\)。在给定相似阈值 \(\varepsilon\) 时，模板集覆盖 GT 的条件为：

\[
\forall G_j\in\mathcal{G},\quad
\exists T_k\in\mathcal{T},\quad
d_{\mathrm{shape}}(G_j,T_k)\leq\varepsilon
\]

最少需要的模板数量是：

\[
N_\varepsilon(\mathcal{G})
=
\min_{\mathcal{C}}
|\mathcal{C}|
\quad
\mathrm{s.t.}\quad
\max_{G_j\in\mathcal{G}}
\min_{C\in\mathcal{C}}
d_{\mathrm{shape}}(G_j,C)
\leq\varepsilon
\]

这正是 metric covering / k-center 一类问题：目标是选择若干中心，使所有对象落在固定半径的球内，而不是让参数矩阵满秩。[metric k-center 的覆盖定义](https://arxiv.org/abs/2007.04059)

因此用户所说的“当前 GT 曲线的秩”，建议在项目中统一改名为：

```text
GT有效形状模态数
或
GT epsilon-cover number
```

避免和线性代数 rank 混淆。

---

## 3. 如何定义“两个曲线形状相似”

### 3.1 不比较绝对位置

GT 和模板在相同 \(y\) anchors 上采样。对 GT 的合法 support，先求最优 \(x/z\) 平移：

\[
b^*
=
\arg\min_b
\sum_{i\in V}
\left\|
(T_i+b)_{xz}-(G_i)_{xz}
\right\|_2^2
\]

平方误差下有闭式解：

\[
b^*
=
\frac{1}{|V|}
\sum_{i\in V}
\left((G_i)_{xz}-(T_i)_{xz}\right)
\]

随后计算平移对齐残差：

\[
r_i=
\left\|
(T_i+b^*)_{xz}-(G_i)_{xz}
\right\|_2
\]

这样：

- 左右位置不同但形状相同的平行车道距离接近 0；
- 近处相同、远处转向不同的车道残差会逐渐增大；
- 不会要求 Shape Prior 直接预测最终车道绝对位置。

### 3.2 同时比较一阶和二阶几何

仅比较离散点平均距离可能掩盖局部转向变化，建议同时使用：

\[
\theta_i
=
\operatorname{atan2}(\Delta x_i,\Delta y_i)
\]

\[
\phi_i
=
\operatorname{atan2}(\Delta z_i,\Delta y_i)
\]

\[
\kappa_i
\approx
\frac{\Delta\theta_i}{\Delta y_i},
\qquad
\kappa^z_i
\approx
\frac{\Delta\phi_i}{\Delta y_i}
\]

它们分别对应：

- 横向 heading/yaw；
- 纵向 grade/pitch；
- 横向曲率变化；
- 纵向曲率变化。

用导数描述曲线天然消除了平移。弹性曲线分析中的 SRV/SRVF 表示同样从曲线速度出发建立平移不变的形状表示。[Srivastava 等，Shape Analysis of Elastic Curves](https://doi.org/10.1109/TPAMI.2010.184)

当前项目已经使用 yaw、pitch、curvature、vertical curvature，这一方向是合理的；建议增加“最优平移后的几何 residual”，让“0.3 m 相似”具有直接物理意义。

### 3.3 不建议当前直接使用 Chamfer、Fréchet 或 Soft-DTW

- Chamfer 不保留相同 \(y\) 的语义对应，可能让远处点错误匹配近处点；
- 离散 Fréchet 距离保留曲线点序关系，适合不同采样序列之间的曲线比较。[Eiter、Mannila，Discrete Fréchet Distance](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf)
- Soft-DTW 是可微的序列对齐损失，适合采样速度或序列长度不同的时间序列。[Cuturi、Blondel，Soft-DTW](https://proceedings.mlr.press/v70/cuturi17a.html)

但 LaneLoom 的 \(y\) 是车辆坐标中的真实前向距离，`y=20m` 不应该通过动态时间规整匹配到 `y=40m`。当前已经有固定且物理对齐的 \(y\) anchors，因此 pointwise + tangent/curvature distance 更合适。

---

## 4. 推荐的形状距离

训练用连续距离：

\[
\begin{aligned}
d_{\mathrm{shape}}(T,G)
=&
w_r\,\operatorname{mean}
\rho\left(\frac{r_i}{\sigma_r}\right)\\
&+
w_\theta\,\operatorname{mean}
\rho\left(\frac{\operatorname{wrap}
(\theta_i^T-\theta_i^G)}
{\sigma_\theta}\right)\\
&+
w_\phi\,\operatorname{mean}
\rho\left(\frac{\phi_i^T-\phi_i^G}
{\sigma_\phi}\right)\\
&+
w_\kappa\,\operatorname{mean}
\rho\left(\frac{\kappa_i^T-\kappa_i^G}
{\sigma_\kappa}\right)\\
&+
w_{\kappa z}\,\operatorname{mean}
\rho\left(\frac{\kappa_i^{z,T}-\kappa_i^{z,G}}
{\sigma_{\kappa z}}\right)
\end{aligned}
\]

其中 \(\rho\) 使用 Smooth L1/Huber，所有项只在 GT 合法 support 内计算。

建议第一版尺度：

```text
sigma_r         = 0.30 m
sigma_yaw       = 1.0 degree
sigma_pitch     = 0.5 degree
sigma_curvature = 0.0015 1/m
sigma_vcurve    = 0.00075 1/m

weights:
residual        = 0.50
yaw             = 0.20
pitch           = 0.10
curvature       = 0.10
vertical_curve  = 0.10
```

这些不是通用数学常数，而是适合当前 0–100 m、2 m anchor 间隔以及 Q0 0.3 m 覆盖目标的初始超参数，最终应通过训练集 GT pair audit 校准。

### 4.1 用于审计和聚类的硬相似条件

建议两个曲线被认为可以共用一个 Shape Template，当且仅当：

```text
共同合法 anchor >= 4
且 aligned residual 的 P80 <= 0.30 m
且 mean yaw error <= 1.0 degree
且 mean pitch error <= 0.5 degree
```

曲率项用于训练梯度和长尾诊断，不建议初版再叠加过多硬阈值。

为什么使用 `P80 <= 0.30m`：

- 与当前 Q0 coverage 的 `80% anchors within 0.3m` 定义一致；
- 允许少数 GT 插值点或端点噪声；
- 比仅看 mean 更不容易被“多数点很好、局部急弯完全错误”掩盖。

在 20 m support 上，1° 的持续 heading 误差会产生约：

\[
20\tan(1^\circ)\approx0.35m
\]

所以 1° yaw 阈值和 0.3 m 几何阈值处在一致量级。更长 support 由 P80 residual 继续收紧。

### 4.2 短 GT 的处理

短 GT 只提供局部形状证据。建议：

- 少于 4 个合法 anchor：不参与“新增形状模态”的 GT 聚类；
- 仍可参与局部 lane coverage，但权重按 support span 截断：

\[
w_j=\min\left(1,\frac{\mathrm{span}_y(G_j)}{20m}\right)
\]

这里的 GT visibility/support 只用于训练 loss mask，不进入 LiDAR-BEV 前向推理，因此属于监督信号，不构成推理阶段的 GT 泄漏。

---

## 5. 先从 GT 得到当前样本真正需要的模态数

先用上述硬相似条件计算 GT 两两距离，然后使用 complete-link 聚类：

```text
同一簇内任意两条 GT 都必须满足相似阈值
```

不应使用 single-link connected component，因为可能出现：

```text
A 与 B 相似
B 与 C 相似
A 与 C 不相似
```

却把三者全部合并的问题。

每个 cluster 选择总距离最小的真实 GT 作为 medoid。k-medoids 的中心本来就是数据集内具有代表性的真实对象，因此适合使用任意曲线 dissimilarity，而不要求计算欧氏均值。[Kaufman、Rousseeuw，PAM/k-medoids](https://doi.org/10.1002/9780470316801.ch2)

得到：

\[
\mathcal{C}_{GT}=\{C_1,\ldots,C_M\},
\qquad
M=N_\varepsilon(\mathcal{G})
\]

其中：

- \(M=1\)：例如多条近似平行直行车道；
- \(M=2\)：例如直行 + 右转；
- \(M=3\)：例如直行 + 左转 + 右转；
- \(M>5\)：说明当前 5 模板上限在既定相似阈值下理论上无法完全覆盖，应记录 `diag_gt_shape_mode_overflow`，不能通过增加 loss 权重解决。

---

## 6. 推荐损失

### 6.1 每条 GT 的 coverage loss

记：

\[
D_{jk}=d_{\mathrm{shape}}(G_j,T_k)
\]

最直接的 facility-location / nearest-center 目标是：

\[
L_{\mathrm{lane-cover}}
=
\frac{\sum_j w_j\min_k \rho(D_{jk})}
{\sum_j w_j}
\]

`min` 在几乎所有位置都有次梯度，PyTorch 可以直接反向传播。

如果训练初期希望多个模板获得梯度，可使用温度退火的 log-sum-exp SoftMin：

\[
\operatorname{SoftMin}_\tau(D_j)
=
-\tau\log\sum_k
\exp(-D_{jk}/\tau)
\]

它对每个距离的导数都是非负 soft assignment：

\[
\frac{\partial\operatorname{SoftMin}}
{\partial D_{jk}}
=
\operatorname{softmax}_k(-D_{jk}/\tau)
\]

因此不会给非最佳模板产生“把距离变大”的反向梯度。但普通 SoftMin 会因为多个模板同时接近而额外下降，所以必须配合后面的 mode matching / diversity，并把温度逐渐降小。

建议：

```text
epoch 1～2: tau = 0.20
epoch 3～6: tau 线性降到 0.05
之后: hard min 或 tau = 0.03
```

### 6.2 GT 模态与模板的一对一匹配

只对 \(M\) 个 GT medoids 和 5 个模板构造距离矩阵：

\[
D^{mode}_{mk}
=
d_{\mathrm{shape}}(C_m,T_k)
\]

用 Hungarian 得到注入映射：

\[
\pi:\{1,\ldots,M\}\rightarrow\{1,\ldots,5\}
\]

然后：

\[
L_{\mathrm{mode-match}}
=
\frac1M
\sum_{m=1}^{M}
\rho(D^{mode}_{m,\pi(m)})
\]

这一项直接保证：

- 两个明显不同的 GT 模态不能都只由同一个模板负责；
- 如果预测模板全部坍缩，而 GT 存在多个模式，则至少部分一对一匹配距离会很大；
- 如果多个平行 GT 形状相同，它们先聚为一个 mode，不会错误要求多个模板。

### 6.3 只约束“有效模板”的 diversity

有效模板集合为：

\[
A=\{T_{\pi(1)},\ldots,T_{\pi(M)}\}
\]

简单而稳健的 margin loss：

\[
L_{\mathrm{matched-diversity}}
=
\frac{2}{M(M-1)}
\sum_{a<b}
\left[
\varepsilon_{\mathrm{sep}}
-d_{\mathrm{shape}}(A_a,A_b)
\right]_+^2
\]

建议：

```text
epsilon_sep = GT similarity threshold
```

即：既然两个模板分别负责两个不同 GT mode，那么它们彼此也不能仍被判断为“同一种形状”。

注意只对 matched active templates 施加该项：

- 当 \(M=1\) 时 diversity loss 为 0；
- 当 \(M=3\) 时只要求这 3 个有效模板分离；
- 不会强迫单一直路场景凭空生成另外 4 个极端形状。

### 6.4 可选：DPP/log-det diversity

DPP 用核矩阵的行列式衡量集合多样性；当特征向量重复或线性依赖时，其张成体积趋近 0。DPP 还可以把每个对象的 quality 和集合 diversity 分开建模。[Kulesza、Taskar，DPP for Machine Learning](https://www.alexkulesza.com/pubs/dpps_fnt12.pdf)

对有效模板构造平移不变、归一化后的形状 embedding \(\varphi(A_a)\)，再定义 RBF Gram kernel：

\[
S_{ab}
=
\exp\left(
-\frac{\|\varphi(A_a)-\varphi(A_b)\|_2^2}
{2\sigma^2}
\right)
\]

可使用：

\[
L_{\mathrm{DPP}}
=
-\frac1M
\log\det(S+\delta I)
\]

推荐：

```text
delta = 1e-4
```

重复模板使 \(S\) 接近奇异，loss 增大。

但第一版更建议使用 pairwise margin：

- 更容易解释阈值；
- 数值更稳定；
- 不依赖形状 embedding 的核设计；
- 5 个模板下计算量没有问题。

可以把 DPP 作为后续 ablation，而不是第一版主损失。

### 6.5 防止某个 Shape Token 永久死亡

仅做逐样本 Hungarian 时，某些 token 可能长期没有被选中。可以统计一个 batch 或 EMA 窗口内的匹配频率：

\[
p_k=
\frac{n_k+\epsilon}
{\sum_l(n_l+\epsilon)}
\]

加入很弱的 usage entropy：

\[
L_{\mathrm{usage}}
=
\log 5
-H(p)
\]

这与在线聚类中防止 prototype assignment 坍缩的思想一致；例如 SwAV 使用带约束的在线 assignment，使 prototype 得到较均衡使用。[Caron 等，SwAV](https://arxiv.org/abs/2006.09882)

但道路形状分布本身并不均衡，直行显著多于急弯，所以该权重必须很小：

```text
lambda_usage = 1e-3 或更小
```

它只用于防止 token 完全死亡，不能强迫五种模式出现频率相同。

---

## 7. 总损失建议

第一版推荐：

\[
\begin{aligned}
L_{\mathrm{shape}}
=&
1.0L_{\mathrm{lane-cover}}\\
&+
1.0L_{\mathrm{mode-match}}\\
&+
0.05L_{\mathrm{matched-diversity}}\\
&+
0.001L_{\mathrm{usage}}
\end{aligned}
\]

其中：

- `lane-cover` 保证每条 GT 有模板负责；
- `mode-match` 保证不同 GT 形状模式分配给不同模板；
- `matched-diversity` 明确惩罚有效模板重复；
- `usage` 只防止长期死 token。

当前形式的：

\[
\sum_k
\frac1{\sum_l \exp(-d(T_k,T_l)/\tau)}
\]

最小化时倾向于把全部模板合并，因此更接近“压缩有效模板数量”，但它：

- 无法真正减少固定的 5×101 条 query；
- 可能制造大量重复 Q0；
- 不知道当前 GT 实际需要几个 mode；
- 与用户现在提出的“有效模板不应线性相关”目标方向相反。

所以建议把当前 `loss_lidar_shape_compact` 替换为上述 `mode-match + matched-diversity`，而不是简单把 compact loss 权重重新打开。

---

## 8. 固定 5 个模板时，到底应该强制 rank=5 吗？

明确答案：

> 不应该。5 是容量上限，有效模板数应匹配当前样本的 GT 有效形状模态数 \(M_{\mathrm{GT}}\)，而不是固定等于 5。

分类讨论：

### 情况 A：GT 只有 1 个形状 mode

```text
多条车道，但都是平移后的近似直行
M_GT = 1
```

正确目标：

- 至少一个模板准确覆盖；
- 不要求另外 4 个模板在该样本上彼此独立。

### 情况 B：GT 有 3 个形状 mode

```text
直行 + 左转 + 右转
M_GT = 3
```

正确目标：

- 3 个 mode 一对一匹配 3 个模板；
- 这 3 个 matched templates 不能重复；
- 剩余 2 个模板不参与该样本的 diversity 强制。

### 情况 C：GT 有 5 个形状 mode

此时若 5 个模板只有 3 个有效形状，`mode-match` 会让至少两个 GT medoid 匹配到错误模板，coverage loss 自然增大；`matched-diversity` 也会惩罚重复。

### 情况 D：GT 有超过 5 个形状 mode

这是结构容量不足：

```text
M_GT > max_templates
```

任何 rank loss 都不能让 5 个 nearest-template centers 在固定误差阈值下覆盖超过容量的互异模式。应该：

- 记录比例；
- 检查阈值是否过严；
- 如果比例不可忽略，再增加 `max_templates`。

---

## 9. 必须增加的诊断

建议训练日志增加：

```text
diag_gt_shape_mode_count_mean
diag_gt_shape_mode_count_p95
diag_gt_shape_mode_overflow_rate

diag_shape_lane_cover_recall_030
diag_shape_mode_cover_recall_030
diag_shape_best_p80_residual

diag_matched_template_pair_min_distance
diag_matched_template_duplicate_count

diag_shape_token_usage_0 ... usage_4
diag_dead_shape_token_count

diag_template_parameter_effective_rank
diag_template_shape_kernel_logdet
```

其中 effective rank 和 log-det 应明确标注为诊断：

- 它们可以发现数值坍缩；
- 不能独立证明 GT 已被覆盖；
- 也不能替代 `shape_mode_cover_recall`。

---

## 10. 推荐实施顺序

1. 复用当前 yaw/pitch/curvature 描述，加入最优平移后的 point residual。
2. 在数据审计脚本中扫描 GT 两两 shape distance。
3. 根据分布确认 `P80<=0.3m`、yaw 1°、pitch 0.5° 是否合理。
4. complete-link 聚类 GT，生成每个 sample 的 GT shape medoids。
5. 实现 `lane-cover + medoid Hungarian mode-match`。
6. 加入仅对 matched templates 生效的 pairwise diversity。
7. 记录 mode overflow、coverage 和 token usage。
8. 只有确认仍发生 matched-template 坍缩后，再尝试 DPP/log-det ablation。

---

## 参考资料

1. Olivier Roy, Martin Vetterli. [The Effective Rank: A Measure of Effective Dimensionality](https://infoscience.epfl.ch/entities/publication/26352894-e3af-4922-aea4-8aab418b0358). EUSIPCO, 2007.
2. Michal Aharon, Michael Elad, Alfred Bruckstein. [K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation](https://doi.org/10.1109/TSP.2006.881199). IEEE Transactions on Signal Processing, 2006.
3. Alex Kulesza, Ben Taskar. [Determinantal Point Processes for Machine Learning](https://www.alexkulesza.com/pubs/dpps_fnt12.pdf). Foundations and Trends in Machine Learning, 2012.
4. Anuj Srivastava et al. [Shape Analysis of Elastic Curves in Euclidean Spaces](https://doi.org/10.1109/TPAMI.2010.184). IEEE TPAMI, 2011.
5. Thomas Eiter, Heikki Mannila. [Computing Discrete Fréchet Distance](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf). Technical Report CD-TR 94/64, 1994.
6. Marco Cuturi, Mathieu Blondel. [Soft-DTW: a Differentiable Loss Function for Time-Series](https://proceedings.mlr.press/v70/cuturi17a.html). ICML, 2017.
7. Leonard Kaufman, Peter Rousseeuw. [Partitioning Around Medoids](https://doi.org/10.1002/9780470316801.ch2). Finding Groups in Data, 1990.
8. Mathilde Caron et al. [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882). NeurIPS, 2020.

