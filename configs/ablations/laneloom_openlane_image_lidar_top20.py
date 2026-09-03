"""Single-frame OpenLane ablation with a 32-query refinement budget."""

_base_ = ['../main/laneloom_openlane_image_lidar.py']

work_dir = './work_dirs/laneloom_openlane_image_lidar_top20'

model = dict(
    lane_head=dict(
        lane_decoder=dict(layer_query_budget=32)))
