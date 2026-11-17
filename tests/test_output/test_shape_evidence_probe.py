"""Interventions must preserve point identity and evaluate unordered templates."""
import torch

from LaneLoom.tools.probe_shape_evidence import intervene, matched_change


def test_geometric_intervention_carries_rgb_and_queue_with_each_point():
    points = torch.tensor([[1., 20., .1, .5, .2, .3, .4, .5, 0.],
                           [2., 60., .2, .6, .1, .6, .7, .8, 2.]])
    transformed = intervene(points, 'bend_right')
    torch.testing.assert_close(transformed[:, 0], points[:, 0] + 5 * (points[:, 1] / 100).square())
    torch.testing.assert_close(transformed[:, 1:], points[:, 1:])
    torch.testing.assert_close(intervene(points, 'far_only'), points[1:])
    torch.testing.assert_close(intervene(points, 'near_only'), points[:1])
    assert points[0, 0] == 1


def test_template_comparison_matches_permuted_identity_and_known_bend():
    y = torch.tensor([20., 40., 60., 100.])
    base = torch.stack((torch.zeros(4), y, torch.zeros(4)), -1)[None].repeat(2, 1, 1)
    base[1, :, 0] = 10.
    permuted = base.flip(0)
    assert matched_change(base, permuted)['mean_xz_change_m'] == 0
    expected = base.clone()
    expected[:, :, 0] += 5 * (y / 100).square()
    result = matched_change(base, expected.flip(0), expected)
    assert result['expected_error_m'] == 0
    assert result['mean_dx_at_20_40_60_100'] == [.2, .8, 1.8, 5.]
    assert abs(result['nonlinear_x_response_gain'] - 1.) < 1e-5
