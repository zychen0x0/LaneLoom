import numpy as np

from LaneLoom.datasets.openlane.full_shape_targets import (
    build_full_shape_targets,
    _bezier_basis,
    _fit_regularized_cubic,
)


def test_regularization_cannot_straighten_a_known_visible_curve_in_fidelity_mode():
    # Independent analytic geometry: no user target frames or annotations.
    for low, high in ((5, 35), (5, 65), (5, 100), (35, 65)):
        y = np.linspace(low, high, 201)
        values = .003 * y**2
        controls = _fit_regularized_cubic(
            y, values, 100., (-50., 50.), visible_fit_tolerance_m=.01)
        assert np.max(np.abs(_bezier_basis(y, 100.) @ controls - values)) <= .01001


def test_full_target_fidelity_mode_preserves_visible_uphill_bend():
    anchors = np.arange(5., 100.1, 5., dtype=np.float32)
    lane = np.stack((.003 * anchors**2, anchors, .0003 * anchors**2), -1)[None]
    visible = ((anchors >= 5) & (anchors <= 65))[None]
    targets = build_full_shape_targets(
        lane, visible, anchors, visible_fit_tolerance_m=.01)
    assert targets['gt_lanes_full_valid'].tolist() == [True]
    error = targets['gt_lanes_3d_full'][0, visible[0]][:, [0, 2]] - lane[0, visible[0]][:, [0, 2]]
    assert np.max(np.abs(error)) < .02


def test_parallel_lane_fragments_become_one_complete_translation_free_template():
    anchors = np.arange(2.0, 100.1, 2.0, dtype=np.float32)
    lanes = np.zeros((4, len(anchors), 3), dtype=np.float32)
    visibility = np.zeros((4, len(anchors)), dtype=np.float32)
    lanes[:, :, 1] = anchors
    visible = (anchors >= 20.0) & (anchors <= 60.0)
    for lane_index, (x_offset, z_offset) in enumerate(((2.0, 0.3), (5.0, -0.2))):
        lanes[lane_index, :, 0] = x_offset + 0.02 * anchors
        lanes[lane_index, :, 2] = z_offset + 0.01 * anchors
        visibility[lane_index, visible] = 1.0

    targets = build_full_shape_targets(
        lanes, visibility, anchors, max_templates=5)

    assert targets['gt_lanes_3d_full'].shape == lanes.shape
    assert targets['gt_lanes_full_valid'].tolist() == [True, True, False, False]
    assert targets['gt_shape_templates'].shape == (5, len(anchors), 3)
    assert targets['gt_shape_template_valid'].tolist() == [
        True, False, False, False, False]
    assert np.allclose(
        targets['gt_lanes_3d_full'][0, :, 0], 2.0 + 0.02 * anchors,
        atol=2e-3)
    assert np.allclose(
        targets['gt_lanes_3d_full'][0, :, 2], 0.3 + 0.01 * anchors,
        atol=2e-3)
    assert np.allclose(
        targets['gt_shape_templates'][0, :, 0], 0.02 * anchors,
        atol=2e-3)
    assert np.allclose(
        targets['gt_shape_templates'][0, :, 2], 0.01 * anchors,
        atol=2e-3)


def test_distinct_lane_shapes_produce_distinct_complete_templates():
    anchors = np.arange(2.0, 100.1, 2.0, dtype=np.float32)
    lanes = np.zeros((3, len(anchors), 3), dtype=np.float32)
    visibility = np.zeros((3, len(anchors)), dtype=np.float32)
    lanes[:, :, 1] = anchors
    visible = (anchors >= 10.0) & (anchors <= 70.0)
    lanes[0, :, 0] = 0.01 * anchors
    lanes[1, :, 0] = 0.003 * anchors ** 2
    visibility[:2, visible] = 1.0

    targets = build_full_shape_targets(
        lanes, visibility, anchors, max_templates=5)

    template_valid = targets['gt_shape_template_valid']
    unique_shapes = np.unique(np.round(
        targets['gt_shape_templates'][template_valid, :, 0], decimals=3),
        axis=0)
    assert template_valid.sum() == 2
    assert len(unique_shapes) == 2


def test_endpoint_outlier_is_not_forced_into_complete_curve():
    anchors = np.arange(2.0, 100.1, 2.0, dtype=np.float32)
    lanes = np.zeros((1, len(anchors), 3), dtype=np.float32)
    visibility = np.zeros((1, len(anchors)), dtype=np.float32)
    lanes[0, :, 1] = anchors
    lanes[0, :, 0] = 0.02 * anchors
    visible = (anchors >= 20.0) & (anchors <= 60.0)
    visibility[0, visible] = 1.0
    lanes[0, np.flatnonzero(visible)[-1], 0] += 15.0

    targets = build_full_shape_targets(
        lanes, visibility, anchors, max_templates=5)

    assert abs(targets['gt_lanes_3d_full'][0, -1, 0] - 2.0) < 1.0


def test_too_short_fragment_is_not_used_as_a_hard_full_range_target():
    anchors = np.arange(5.0, 100.1, 5.0, dtype=np.float32)
    lanes = np.zeros((1, len(anchors), 3), dtype=np.float32)
    visibility = np.zeros((1, len(anchors)), dtype=np.float32)
    lanes[0, :, 1] = anchors
    lanes[0, :, 0] = 0.02 * anchors
    visibility[0, :2] = 1.0

    targets = build_full_shape_targets(
        lanes, visibility, anchors, max_templates=5)

    assert not targets['gt_lanes_full_valid'].any()
    assert not targets['gt_shape_template_valid'].any()


def test_support_threshold_is_rechecked_after_endpoint_outlier_filtering():
    anchors = np.arange(2.0, 100.1, 2.0, dtype=np.float32)
    lanes = np.zeros((1, len(anchors), 3), dtype=np.float32)
    visibility = np.zeros((1, len(anchors)), dtype=np.float32)
    lanes[0, :, 1] = anchors
    lanes[0, :, 0] = 0.02 * anchors
    visible_indices = np.flatnonzero((anchors >= 20.0) & (anchors <= 40.0))
    visibility[0, visible_indices] = 1.0
    lanes[0, visible_indices[:2], 0] += 15.0
    lanes[0, visible_indices[-2:], 0] -= 15.0

    targets = build_full_shape_targets(
        lanes, visibility, anchors, max_templates=5,
        min_lane_points=4, min_lane_span=15.0)

    assert not targets['gt_lanes_full_valid'].any()
    assert not targets['gt_shape_template_valid'].any()


def test_short_lane_borrows_only_from_a_compatible_long_lane():
    anchors = np.arange(5.0, 100.1, 5.0, dtype=np.float32)
    lanes = np.zeros((2, len(anchors), 3), dtype=np.float32)
    lanes[:, :, 1] = anchors
    lanes[0, :, 0] = 2.0 + 0.01 * anchors
    lanes[1, :, 0] = 5.0 + 0.01 * anchors
    lanes[0, :, 2] = 0.3 + 0.015 * anchors
    lanes[1, :, 2] = -0.2 + 0.015 * anchors
    visibility = np.zeros((2, len(anchors)), dtype=np.float32)
    visibility[0, (anchors >= 20) & (anchors <= 35)] = 1
    visibility[1, (anchors >= 10) & (anchors <= 100)] = 1

    old = build_full_shape_targets(lanes, visibility, anchors)
    new = build_full_shape_targets(
        lanes, visibility, anchors, guarded_short_merge=True,
        z_extrapolation_grace_m=20.0, z_extrapolation_decay_m=20.0)

    assert old['gt_shape_template_valid'].sum() == 2
    assert new['gt_shape_template_valid'].sum() == 1
    assert np.allclose(new['gt_lanes_full_z_weights'][0],
                       new['gt_lanes_full_z_weights'][1])
    assert np.allclose(new['gt_shape_template_z_weights'][0], 1.0)
    assert np.all(new['gt_shape_template_z_weights'][1:] == 0)

    weak_visibility = visibility.copy()
    weak_visibility[1, anchors > 45] = 0
    weak_donor = build_full_shape_targets(
        lanes, weak_visibility, anchors, guarded_short_merge=True)
    assert weak_donor['gt_shape_template_valid'].sum() == 2

    divergent_lanes = lanes.copy()
    divergent_lanes[1, :, 2] += 0.05 * anchors
    divergent = build_full_shape_targets(
        divergent_lanes, visibility, anchors, guarded_short_merge=True)
    assert divergent['gt_shape_template_valid'].sum() == 2


def test_short_singleton_z_weight_fades_only_beyond_visible_support():
    anchors = np.arange(5.0, 100.1, 5.0, dtype=np.float32)
    lanes = np.zeros((1, len(anchors), 3), dtype=np.float32)
    lanes[0, :, 1] = anchors
    lanes[0, :, 2] = 0.01 * anchors
    visibility = np.zeros((1, len(anchors)), dtype=np.float32)
    visibility[0, (anchors >= 20) & (anchors <= 35)] = 1

    targets = build_full_shape_targets(
        lanes, visibility, anchors, guarded_short_merge=True,
        z_extrapolation_grace_m=20.0, z_extrapolation_decay_m=20.0)
    weights = targets['gt_shape_template_z_weights'][0]

    assert np.allclose(weights[(anchors >= 5) & (anchors <= 55)], 1.0)
    assert np.isclose(weights[anchors == 75], np.exp(-1.0))
    assert np.isclose(weights[anchors == 95], np.exp(-2.0))
    assert np.array_equal(weights, targets['gt_lanes_full_z_weights'][0])
