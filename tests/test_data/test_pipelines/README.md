# OpenLane Pipeline Tests


## 当前 random_aug 网络输入

```bash
cd /workspace/LaneLoom

SAMPLE=/workspace/LaneLoom/data/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg

PYTHONPATH=/workspace \
LANELOOM_VIS_INPUT=1 \
LANELOOM_VIS_IMAGE_PATH="$SAMPLE" \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_one_openlane_network_input_sample \
-q -s
```

当前训练配置关闭 GridMask。`openlane_resized_vs_network_input.jpg` 对比
左侧三帧 resize 原图与右侧三帧 `random_aug` 网络输入；
`openlane_network_input_sample.jpg` 保存同一组三帧增强结果。

## 单项图像增强效果对比

下面的测试读取同一张图片，将训练配置中的 11 种 Albumentations 增强分别
强制执行一次。输出为 `原图 + 11 种单项增强` 的 3 列 × 4 行对比图，不会
修改训练 pipeline 中的概率或组合逻辑。

```bash
cd /workspace/LaneLoom

SAMPLE=/workspace/LaneLoom/data/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg

PYTHONPATH=/workspace \
LANELOOM_VIS_INDIVIDUAL_AUGS=1 \
LANELOOM_VIS_IMAGE_PATH="$SAMPLE" \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_individual_openlane_augmentations \
-q -s
```

默认输出路径是
`tests/test_data/test_pipelines/show_load_img/openlane_individual_augmentations.jpg`。
测试固定了随机种子，方便重复对比；每个单项的 `p` 仅在这个可视化测试中
临时设为 `1.0`。

训练时并不是从全部 11 种增强中只选一种。5 个顶层配置会按照声明顺序分别
进行独立概率判断，因此一次输入可能不增强、只增强一次，或者叠加多个增强。
只有 `OneOf` 组内部是单选：当该组触发时，最多选择其中一个子增强。

## multi_aug_views 效果对比

下面的测试将同一张图片分别应用到当前配置的 `original`、`brightness`、
`fog`、`blur`、`occlusion` 和 `random` 六个 view，输出 3 列 × 2 行对比图。
测试临时启用 `view_mode='multi_aug_views'` 和 `num_views=6`，不会修改训练配置。

```bash
cd /workspace/LaneLoom

SAMPLE=/workspace/LaneLoom/data/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg

PYTHONPATH=/workspace \
LANELOOM_VIS_MULTI_AUG_VIEWS=1 \
LANELOOM_VIS_IMAGE_PATH="$SAMPLE" \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_openlane_multi_aug_views \
-q -s
```

默认输出路径是
`tests/test_data/test_pipelines/show_load_img/openlane_multi_aug_views.jpg`。
每个 view 使用固定随机种子，因此随机参数和 `OneOf` 的选择可以稳定复现。

每个外层配置对应一个 view。同一 view 的 `transforms` 会按顺序执行，例如
`fog` 会依次叠加雾和亮度/对比度调整，`blur` 会依次叠加雨和高斯模糊。
`random` 中第一个 `OneOf(p=1.0)` 必定选择一种颜色增强，第二个
`OneOf(p=0.65)` 有 65% 概率再选择压缩、噪声或 ToneCurve 中的一种。

## 当前帧 3D GT 投影

该可视化使用原始图像，只执行 Resize 和 Normalize，不执行 Albu 或 GridMask。

```bash
cd /workspace/LaneLoom

SAMPLE=/workspace/LaneLoom/data/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg

PYTHONPATH=/workspace \
LANELOOM_VIS_3D_GT=1 \
LANELOOM_VIS_IMAGE_PATH="$SAMPLE" \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_openlane_network_input_with_3d_gt \
-q -s
```

## 时序 GT 增强对比

该可视化使用原始图像，只执行 Resize 和 Normalize，不执行 Albu 或 GridMask。

```bash
cd /workspace/LaneLoom

SAMPLE=/workspace/LaneLoom/data/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg

PYTHONPATH=/workspace \
LANELOOM_VIS_TEMPORAL_GT=1 \
LANELOOM_TEMPORAL_GT_IMAGE_PATH="$SAMPLE" \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_temporal_gt_enhancement_comparison \
-q -s
```

这些环境变量用于让默认跳过的可视化测试显式运行，并选择一个确定的真实样本。
三个测试的输出路径已经内置，通常不需要再传 `*_OUTPUT`。


## 长尾 GT 表述与 Q0 覆盖审计

下面的测试会同时输出：原始 JSON GT（绿色）、固定 `y=0:2:100` 的训练
GT（橙色），以及使用 GT 拟合出的理想单模态 Q0（灰色）。默认包含 U 型、
横向线、停止线、多方向道路和非规则曲线等 6 个文档样本。

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace \
LANELOOM_AUDIT_GT=1 \
LANELOOM_GT_AUDIT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_gt_q0_audit.jpg \
LANELOOM_GT_AUDIT_REPORT=tests/test_data/test_pipelines/show_load_img/openlane_gt_q0_audit.json \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_audit_openlane_gt_representation_and_q0_coverage \
-q -s
```

用 `LANELOOM_GT_AUDIT_IMAGES`（Linux 下用 `:` 分隔）可以替换默认样本。
如果希望把“最近理想 Q0 平均距离超过 1 m”作为失败判据，再加：

```bash
LANELOOM_GT_AUDIT_MAX_Q0_DISTANCE=1.0
```


## GT 三次 Bézier 二次拟合可视化

下面的测试使用一个确定性真实样本，同时绘制原始 GT、第一次全点拟合、
基于 `0.75m` residual 的 inlier 筛选，以及被稳定性检查拒绝的第二次病态
拟合。输出包含相机投影、局部/全局 BEV、竖向曲线和设计矩阵条件数。

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace \
LANELOOM_VIS_PATHOLOGICAL_GT=1 \
LANELOOM_PATHOLOGICAL_GT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_pathological_bezier_refit.jpg \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_pathological_openlane_bezier_refit \
-q -s
```

下面的真实样本展示另一种情况：inlier 覆盖完整、设计矩阵稳定，因此第二次
拟合被接受，并把保留点上的 RMSE 从约 `0.144m` 降到 `0.013m`：

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace \
LANELOOM_VIS_BENEFICIAL_REFIT=1 \
LANELOOM_BENEFICIAL_REFIT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_beneficial_bezier_refit.jpg \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_beneficial_openlane_bezier_refit \
-q -s
```
