# OpenLane Pipeline Tests


## 多视图可增强视化  &&  时序GT增强可视化


cd /workspace/LaneFormer


DATA_ROOT="${LANEFORMER_DATA_ROOT:-/workspace/lanedet3d/data}"
SAMPLE="$DATA_ROOT/openlane/images/training/segment-8603916601243187272_540_000_560_000_with_camera_labels/155250133677391100.jpg"

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


