"""Second controlled experiment: visible Q0 supervision without early pruning."""

_base_ = ['./laneloom_openlane_q0_visible_coverage.py']

work_dir = './work_dirs/laneloom_openlane_q0_visible_coverage_no_prune'
model = dict(lane_head=dict(lane_decoder=dict(layer_query_budget=None)))
