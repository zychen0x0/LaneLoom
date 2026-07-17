# OpenLane Pipeline Tests


## 多视图可增强视化  &&  时序GT增强可视化


cd /workspace/LaneFormer


DATA_ROOT="${LANEFORMER_DATA_ROOT:-/workspace/lanedet3d/data}"
SAMPLE="$DATA_ROOT/openlane/images/training/segment-16534202648288984983_900_000_920_000_with_camera_labels/155934387138996700.jpg"

PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}" \
LANEFORMER_VIS_INPUT=1 \
LANEFORMER_VIS_3D_GT=1 \
LANEFORMER_VIS_TEMPORAL_GT=1 \
LANEFORMER_VIS_IMAGE_PATH="$SAMPLE" \
LANEFORMER_TEMPORAL_GT_IMAGE_PATH="$SAMPLE" \
LANEFORMER_VIS_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_network_input_sample.jpg \
LANEFORMER_VIS_3D_GT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_network_input_3d_gt_bezier.jpg \
LANEFORMER_TEMPORAL_GT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_temporal_gt_bezier.jpg \
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
cd /workspace/LaneFormer

PYTHONPATH=/workspace \
LANEFORMER_AUDIT_GT=1 \
LANEFORMER_GT_AUDIT_OUTPUT=tests/test_data/test_pipelines/show_load_img/openlane_gt_q0_audit.jpg \
LANEFORMER_GT_AUDIT_REPORT=tests/test_data/test_pipelines/show_load_img/openlane_gt_q0_audit.json \
python -m pytest \
tests/test_data/test_pipelines/test_openlane_pipeline.py::test_audit_openlane_gt_representation_and_q0_coverage \
-q -s
```

用 `LANEFORMER_GT_AUDIT_IMAGES`（Linux 下用 `:` 分隔）可以替换默认样本。
如果希望把“最近理想 Q0 平均距离超过 1 m”作为失败判据，再加：

```bash
LANEFORMER_GT_AUDIT_MAX_Q0_DISTANCE=1.0
```

