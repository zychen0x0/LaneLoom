"""Minimal CQG supervision ablation with checkpoint-compatible queries."""

_base_ = ['./laneloom_openlane_image_lidar_temporal.py']

work_dir = './work_dirs/laneloom_openlane_image_lidar_temporal_improve'

model = dict(
    lane_head=dict(
        anchor_prior=dict(
            # Preserve the checkpoint's original unbounded slope
            # parameterization. Q0 geometry below is the only direct CQG
            # supervision in this ablation.
            supervise_shapes=False),
        proposal_expander=dict(
            # Geometry coverage is many-to-one: disjoint GT fragments may
            # jointly supervise the same full-road query. Instance assignment
            # remains one-to-one in the decoder.
            q0_loss_type='separated_pointwise',
            q0_assignment='per_gt_min'),
        # Replace the existing pointwise visibility loss. Visible and
        # invisible anchors contribute equally, preventing short intervals
        # from winning merely because most of the 50 anchors are invisible.
        loss_module=dict(
            visibility_loss_type='balanced_bce')))
