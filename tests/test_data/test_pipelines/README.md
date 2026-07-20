# OpenLane Pipeline Tests


## 多视图可增强视化  &&  时序GT增强可视化


cd /workspace/LaneLoom


DATA_ROOT="${LANELOOM_DATA_ROOT:-/workspace/lanedet3d/data}"
SAMPLE="$DATA_ROOT/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg"

PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}" \
LANELOOM_VIS_INPUT=1 \
LANELOOM_VIS_3D_GT=1 \
LANELOOM_VIS_TEMPORAL_GT=1 \
LANELOOM_VIS_IMAGE_PATH="$SAMPLE" \
LANELOOM_TEMPORAL_GT_IMAGE_PATH="$SAMPLE" \
LANELOOM_VIS_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_network_input_sample.jpg \
LANELOOM_VIS_3D_GT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_network_input_3d_gt_bezier.jpg \
LANELOOM_TEMPORAL_GT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_temporal_gt_bezier.jpg \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_one_openlane_network_input_sample \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_openlane_network_input_with_3d_gt \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_visualize_temporal_gt_enhancement_comparison \
-q -s


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
