# dv-3dlane 的 OpenLane 评测与场景划分

调研对象为 `JMoonr/dv-3dlane` 当前默认分支的提交
[`6155d71`](https://github.com/JMoonr/dv-3dlane/tree/6155d714f2df21856ca0b388edb7555a18c322d7)。以下结论只描述仓库中可验证的实现，不把论文表格反推成代码行为。

## 结论摘要

1. dv-3dlane **不从 GT lane 的几何形状推导 `straight` / `turn`**。仓库里没有这种二分类逻辑，也没有 `straight` 子集。
2. 它把 OpenLane 已经组织好的测试目录当作不同数据集 split：`test/curve_case`、`test/night_case`、`test/up_down_case`、`test/extreme_weather_case`、`test/intersection_case`、`test/merge_split_case`。也就是说，场景归属来自 OpenLane 数据发布方提供的场景子集，而不是 evaluator 看过 GT 后临时分类。
3. 各场景使用同一个 `LaneEval` 和同一套阈值。要得到某一场景结果，本质上是把 `val_split` 指向相应目录并单独运行评测，而不是一次 overall 评测内部按 tag 聚合。
4. 发布配置默认只评 `validation`，并不会自动循环六个 test case。仓库虽然定义了六个场景 split，但默认 `val_split` 是 `OpenLaneSplit.Val`。
5. dv-3dlane 的几何协议与 LaneFormer 配置中的 `0.5 m / 0.75 / 40 m` 有直接对应关系，但实现细节不能概括成“一个配对必须同时通过两个 0.75 才成为唯一 TP”：原实现分别累计 recall 命中和 precision 命中，最后由全数据集计数计算 F1。

## 场景子集到底从哪里来

dv-3dlane 在 [`experiments/utils.py`](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/experiments/utils.py#L11-L21) 中把场景直接声明为目录形式的 split：

| 名称 | 目录 |
|---|---|
| UpDown | `test/up_down_case` |
| Curve | `test/curve_case` |
| ExWeather | `test/extreme_weather_case` |
| Intersect | `test/intersection_case` |
| MS | `test/merge_split_case` |
| Night | `test/night_case` |

`get_dataset` 将选中的 split 字符串直接交给 `LaneDataset`（[`experiments/utils.py` L30-L46](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/experiments/utils.py#L30-L46)），而 `LaneDataset` 从 `data_label_dir/<split>/` 递归读取 JSON（[`data/Load_Data.py` L99-L100, L184-L186](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/data/Load_Data.py#L99-L100)）。因此代码使用的是数据集目录已经表达的场景成员关系。

这与 OpenLane 自己的数据说明一致：官方数据文档列出了上述六个 test case 目录和对应 txt 名单，并说明这些 special-theme segments 从 validation 中选出（[OpenLane 官方 data README](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md#L373-L435)）。不过，对 dv-3dlane 这条 lane evaluation 代码路径，更准确的说法是“读取官方预划分的 case 目录”，而不是“逐帧读取 scene-tag 字段再动态过滤”。

需要注意，发布配置默认设置为 [`val_split = 'OpenLaneSplit.Val'`](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/config/release_iclr/dv3dlane_openlane1000_base.py#L106-L113)，Runner 只构造这一个 valid dataset（[`experiments/runner.py` L53-L66](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/experiments/runner.py#L53-L66)）。仓库虽声明了 `evaluate_case` CLI 参数，但没有代码消费它（[参数定义](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/utils.py#L214-L215)）。所以，若要复现 README 中 All、Curve、Night 等多列结果，需要对整体/各 case 分别选择 split 并运行，或自行写循环，而不是期待默认命令一次输出全部列。

## 每个 split 使用的评测协议

`LaneEval` 的初始化明确设置：

- 在纵向 ROI 内均匀重采样 100 个 y 点；
- `dist_th = 0.5 m`（可由配置覆盖）；
- `ratio_th = 0.75`；
- `close_range = 40 m`。

源码见 [`utils/eval_3D_lane.py` L54-L66](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/eval_3D_lane.py#L54-L66)。这解释了 LaneFormer 配置中 `distance_threshold=0.5`、`match_ratio_threshold=0.75` 和 `near_distance=40` 的数值来源，但 dv-3dlane 是 100 点重采样，不是 LaneFormer 的 51 个固定 anchor。

具体流程如下：

1. GT 先按 visibility 裁剪，GT 和 prediction 再按 y/x ROI 裁剪；只有超过一个可见重采样点的 lane 被保留（[`eval_3D_lane.py` L96-L121, L152-L161](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/eval_3D_lane.py#L96-L121)）。
2. 对每个 GT/pred pair，在共同可见点计算 x-z 欧氏距离；小于 `0.5 m` 的点计为几何命中。只一方可见的位置以 `dist_th` 作为代价，双方都不可见的位置不贡献命中（[`eval_3D_lane.py` L176-L201](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/eval_3D_lane.py#L176-L201)）。
3. 用 min-cost flow 做一对一匹配（不是 SciPy Hungarian，但目标同属二部图最优匹配），并以总 cost 小于 `dist_th × 100` 过滤匹配（[`eval_3D_lane.py` L227-L240](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/eval_3D_lane.py#L227-L240)）。
4. 对保留配对，`matched_points / GT_visible_points ≥ 0.75` 时 recall 计数加一；`matched_points / pred_visible_points ≥ 0.75` 时 precision 计数加一。这两项在代码中是两个独立的 `if`（[`eval_3D_lane.py` L241-L247](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/eval_3D_lane.py#L241-L247)）。
5. 多卡汇总后用总 recall-hit / 总 GT、总 precision-hit / 总 prediction 计算 Recall、Precision 和 F1（[`experiments/manager.py` L105-L136](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/experiments/manager.py#L105-L136)）。近/远 x、z 误差则按 40 m 分段。

预测置信度阈值是另一层过滤。发布配置使用 [`pos_threshold = 0.3`](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/config/release_iclr/dv3dlane_openlane1000_base.py#L22-L25)，评测入口将它作为 `prob_th` 传入，非 GenLaneNet 模型保留最大类别概率高于该阈值的 lane（[`eval_3D_lane.py` L442-L456](https://github.com/JMoonr/dv-3dlane/blob/6155d714f2df21856ca0b388edb7555a18c322d7/utils/eval_3D_lane.py#L442-L456)）。

## LaneFormer 的场景评测实现

LaneFormer 不再根据 GT 几何推导 `straight` / `turn`，也不由 metric 在一个
split 内做场景聚合。`tools/test.py` 会为 `validation` 和六个官方 case 各构造
一份独立配置：将对应目录写入 `dataset_cfg.val_split`，并令 test dataloader
以 `split='val'` 加载它。每次运行的指标均带独立前缀，例如
`OpenLane/curve_case/F1`。

因此，论文对比和实验报告中的场景指标均来自 OpenLane 官方固定子集；不再把
任何 GT 几何启发式当作场景定义。
