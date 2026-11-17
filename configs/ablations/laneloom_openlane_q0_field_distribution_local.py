"""Untrained candidate: preserve different spatial modes during row reading."""
_base_ = ['./laneloom_openlane_q0_field_distribution.py']

work_dir = './work_dirs/q0_bev_holdout/field_distribution_local/supervised'
model = dict(lane_head=dict(anchor_prior=dict(row_query_references=True,
                                              row_query_window_m=1.5)))
