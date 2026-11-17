import numpy as np
import torch
import pytest

from LaneLoom.tools.evaluate_q0_prior import (
    evaluate_controls, summarize_distances, audit_cubic_gt, coverage_goal_report,
    validate_comparison_protocols, evaluate_lane_field, validate_comparison_input_frames)


def test_comparison_requires_identical_input_coordinate_frames_and_roi():
    import copy
    from mmengine.config import Config
    old = Config(dict(model=dict(pts_bev_encoder=dict(point_cloud_range=[-25, 0, -5, 25, 100, 5]))))
    new = copy.deepcopy(old)
    validate_comparison_input_frames(old, new)
    new.model.pts_bev_encoder.align_to_lane_ground = True
    with pytest.raises(ValueError, match='coordinate frame'):
        validate_comparison_input_frames(old, new)
    new = Config(dict(model=dict(pts_bev_encoder=dict(point_cloud_range=[-25, 0, -10, 25, 100, 10]))))
    with pytest.raises(ValueError, match='point-cloud range'):
        validate_comparison_input_frames(old, new)


def test_lane_field_metrics_use_original_point_coordinates_and_do_not_count_as_q0_coverage():
    y = (torch.arange(100) + .5)[None, None, :, None]
    height = (.02*y).expand(1, 1, 100, 50)
    logits = torch.full_like(height, -2.)
    logits[..., 25] = 2.
    points = np.array([[.5, 10., .2], [.5, 20., .4]])
    metrics = evaluate_lane_field(dict(lane_field_logits=logits, lane_height_field=height),
                                  [dict(points=points)], [-25, 0, -5, 25, 100, 5])
    assert metrics['height_mae_m'] < 1e-6
    assert metrics['gt_vs_row_probability_contrast'] > .5
    assert 'covered' not in metrics


def test_lane_field_diagnostic_distinguishes_thin_lane_from_broad_road_response():
    thin = torch.full((1, 1, 100, 100), -2.)
    thin[..., 50] = 2.
    broad = torch.full_like(thin, -2.)
    broad[..., 40:60] = 2.
    points = [dict(points=np.array([[.25, 10., 0.], [.25, 20., 0.]]))]
    a = evaluate_lane_field(dict(lane_field_logits=thin, lane_height_field=torch.zeros_like(thin)),
                            points, [-25, 0, -5, 25, 100, 5])
    b = evaluate_lane_field(dict(lane_field_logits=broad, lane_height_field=torch.zeros_like(broad)),
                            points, [-25, 0, -5, 25, 100, 5])
    assert a['mean_probability_at_gt'] == b['mean_probability_at_gt']
    assert a['fraction_gt_points_near_peak_03'] == 1
    assert b['fraction_gt_points_near_peak_03'] == 0


def test_axis_diagnostics_expose_failure_without_relaxing_joint_coverage():
    points = [dict(json_index=0, points=np.array([[0., 10., 0.], [0., 20., 0.]]))]
    output = evaluate_controls(dict(shape_x_controls=torch.zeros(1, 5, 4),
                                    shape_z_controls=torch.ones(1, 5, 4)), points)
    assert output['paired_xz_p80']['covered'] == 0
    assert output['diagnostic_axis_p80_03']['lateral_only']['covered'] == 1
    assert output['diagnostic_axis_p80_03']['height_only']['covered'] == 0
    ramp = torch.linspace(0, 100, 4)[None, None].expand(1, 5, -1)
    output = evaluate_controls(dict(shape_x_controls=ramp, shape_z_controls=torch.zeros_like(ramp)), points)
    assert output['paired_xz_p80']['covered'] == 0
    assert output['diagnostic_axis_p80_03']['lateral_only']['covered'] == 0
    assert output['diagnostic_axis_p80_03']['height_only']['covered'] == 1


def test_lane_field_ablation_is_explicit_and_cannot_relax_data_or_training_budget():
    old = dict(model_type='LiDARShapeTokenPrior', training_labels_sha256='same',
               batch_size=8, epochs=2, supervision='raw-visible geometry', q0_model_sha256='old')
    new = dict(old, model_type='GuidedSpatialBEVShapePrior', q0_model_sha256='new',
               lane_field_loss_weight=1.,
               supervision='raw-visible geometry; sparse multi-label lane-position and visible-point height field')
    with pytest.raises(ValueError, match='protocols differ'):
        validate_comparison_protocols(old, new)
    validate_comparison_protocols(old, new, allow_lane_field_loss_difference=True)
    for key, value in [('training_labels_sha256', 'different'), ('batch_size', 16), ('epochs', 1),
                       ('supervision', 'completed templates')]:
        changed = dict(new, **{key: value})
        with pytest.raises(ValueError, match='protocols differ'):
            validate_comparison_protocols(old, changed, allow_lane_field_loss_difference=True)
    assert old['model_type'] == 'LiDARShapeTokenPrior'
    changed = dict(new, bev_coordinate_frame='calibrated lane ground')
    with pytest.raises(ValueError, match='protocols differ'):
        validate_comparison_protocols(old, changed, allow_lane_field_loss_difference=True)


def test_goal_requires_all_four_thresholds_all_74_gt_and_strictly_beating_baseline():
    metrics = ('paired_xz_p80', 'paired_xz_p75_03', 'paired_xz_p80_05', 'paired_xz_p75_05')
    samples = []
    remaining = 67
    for index in range(21):
        gt = 3 if index < 20 else 14
        covered = min(gt, remaining)
        remaining -= covered
        samples.append(dict(frame=str(index), raw_gt=gt,
                            baseline={m: dict(gt=gt, covered=0) for m in metrics},
                            spatial={m: dict(gt=gt, covered=covered) for m in metrics}))
    expected = [str(i) for i in range(21)]
    assert coverage_goal_report(dict(samples=samples), expected)['passed']
    samples[0]['spatial']['paired_xz_p80']['covered'] -= 1
    assert not coverage_goal_report(dict(samples=samples), expected)['passed']
    samples[0]['spatial']['paired_xz_p80']['covered'] += 1
    for sample in samples:
        sample['baseline']['paired_xz_p75_05']['covered'] = sample['spatial']['paired_xz_p75_05']['covered']
    assert not coverage_goal_report(dict(samples=samples), expected)['passed']
    with pytest.raises(ValueError, match='original 21 frames'):
        coverage_goal_report(dict(samples=samples[:-1]), expected)


def test_validation_goal_can_require_only_p75_05_without_relaxing_denominator_or_threshold():
    metrics = ('paired_xz_p80', 'paired_xz_p75_03', 'paired_xz_p80_05', 'paired_xz_p75_05')
    sample = dict(frame='validation', raw_gt=266,
                  baseline={m: dict(gt=266, covered=198 if m == 'paired_xz_p75_05' else 0)
                            for m in metrics},
                  spatial={m: dict(gt=266, covered=0) for m in metrics})
    for count, passed in ((238, False), (239, False), (240, True)):
        sample['spatial']['paired_xz_p75_05']['covered'] = count
        result = coverage_goal_report(dict(samples=[sample]), ['validation'], 266,
                                      required_metrics=['paired_xz_p75_05'])
        assert result['passed'] is passed
        assert result['required_metrics'] == ['paired_xz_p75_05']
        assert len(result['criteria']) == 4  # Other diagnostics remain visible.
    assert not coverage_goal_report(dict(samples=[sample]), ['validation'], 266)['passed']
    sample['spatial']['paired_xz_p75_05']['gt'] = 265
    with pytest.raises(ValueError, match='GT lane was dropped'):
        coverage_goal_report(dict(samples=[sample]), ['validation'], 266,
                             required_metrics=['paired_xz_p75_05'])


def test_q0_evaluation_includes_raw_short_gt_omitted_by_fixed_anchors():
    points = np.array([[0., 9.5, 0.], [0., 9.8, 0.], [0., 10.5, 0.]])
    lanes = [dict(json_index=7, points=points)]
    result = evaluate_controls(dict(shape_x_controls=torch.zeros(1, 5, 4),
                                   shape_z_controls=torch.zeros(1, 5, 4)), lanes)
    assert result['paired_xz_p95']['gt'] == 1
    assert result['paired_xz_p95']['covered'] == 1
    assert result['nearest_3d_p95']['covered'] == 1
    assert result['paired_xz_p95']['lanes'][0]['json_index'] == 7


def test_one_candidate_cannot_count_as_two_detectable_gt_lanes():
    # Only query 0 covers either GT. Unique assignment must report one miss.
    distances = [np.array([[.1, .1], [1., 1.]]), np.array([[.1, .1], [2., 2.]])]
    result = summarize_distances(distances, [dict(json_index=0), dict(json_index=1)], .95)
    assert result['gt'] == 2
    assert result['covered'] == 1
    assert len(set(l['query_id'] for l in result['lanes'])) == 2


def test_p75_half_meter_coverage_uses_same_curve_and_strict_distance():
    prediction = dict(shape_x_controls=torch.zeros(1, 5, 4),
                      shape_z_controls=torch.zeros(1, 5, 4))
    points = np.array([[0., 10., .49], [0., 20., .49],
                       [0., 30., .49], [0., 40., 1.]])
    result = evaluate_controls(prediction, [dict(json_index=0, points=points)])
    assert result['paired_xz_p80']['covered'] == 0
    assert result['paired_xz_p75_03']['covered'] == 0
    assert result['paired_xz_p80_05']['covered'] == 0
    assert result['paired_xz_p75_05']['covered'] == 1
    assert result['paired_xz_p75_05']['lanes'][0]['fraction'] == .75
    points[:3, 2] = .5
    result = evaluate_controls(prediction, [dict(json_index=0, points=points)])
    assert result['paired_xz_p75_05']['covered'] == 0
    points[:3, 2] = .29
    result = evaluate_controls(prediction, [dict(json_index=0, points=points)])
    assert result['paired_xz_p75_03']['covered'] == 1
    assert result['paired_xz_p80']['covered'] == 0


def test_raw_cubic_audit_distinguishes_representation_error_from_model_error():
    y = np.linspace(0, 10, 8)
    cubic = np.stack((.001 * y**3, y, .01*y), -1)
    audit = audit_cubic_gt([dict(json_index=0, points=cubic)])[0]
    assert audit['least_squares_p95_xz_m'] < 1e-8
    assert audit['minimum_worst_x_error_any_cubic_m'] < 1e-8
    oscillating = cubic.copy()
    oscillating[:, 0] += np.array([1, -1, 1, -1, 1, -1, 1, -1])
    audit = audit_cubic_gt([dict(json_index=0, points=oscillating)])[0]
    assert audit['p95_requires_all_points']
    assert audit['minimum_worst_x_error_any_cubic_m'] > .3
