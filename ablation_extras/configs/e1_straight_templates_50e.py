"""E1: replace CQG templates with fixed straight lines (everything else = E0).

The 505 initial queries are straight curves at the same 101 lateral seeds.
Template shape losses are switched off because the templates carry no shape
parameters; the initial-query loss and all decoder losses are unchanged.
"""

_base_ = ['./e0_full_50e.py']

work_dir = './work_dirs/ablation_e1_straight_templates_50e'

model = dict(
    lane_head=dict(
        anchor_prior=dict(
            type='AblationStraightTemplatePrior',
            supervise_shapes=False)))
