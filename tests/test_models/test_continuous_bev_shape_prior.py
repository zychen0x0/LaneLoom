"""Physical evidence, continuous association and unchanged Q0 geometry."""
import torch

from LaneLoom.models.backbones.road_evidence import (
    ROAD_EVIDENCE_FIELDS, road_evidence)
from LaneLoom.models.head.continuous_bev_shape_prior import (
    ContinuousBEVShapePrior, continuous_path_probabilities, fit_observed_cubic)
from LaneLoom.models.head.shape_proposal_expander import ShapeProposalExpander


def test_observed_height_color_boundaries_ignore_holes_and_upper_objects():
    points = torch.tensor([
        [.5, .5, 0., 1., 0., .1, .2, .3, 0.],
        [.5, .5, 1., 1., 0., .9, .9, .9, 0.],
        [1.5, .5, .15, 1., 0., .4, .3, .2, 0.],
        [10., .5, .15, 1., 0., .4, .3, .2, 0.],
        [float('nan'), .5, .15, 1., 0., .4, .3, .2, 0.]])
    fields = road_evidence([points], [0, 0, -1, 4, 4, 2], 4, 4)[0]
    index = {name: i for i, name in enumerate(ROAD_EVIDENCE_FIELDS)}
    assert torch.isfinite(fields).all()
    assert fields[index['observed']].sum() == 2
    torch.testing.assert_close(fields[6:9, 0, 0], points[0, 5:8])
    torch.testing.assert_close(fields[index['height_dx'], 0, 1], torch.tensor(.15))
    assert fields[index['color_dx'], 0, 1] > 0
    assert fields[12:, 0, 2:].count_nonzero() == 0  # empty neighbors are not edges
    permuted = road_evidence([points.flip(0)], [0, 0, -1, 4, 4, 2], 4, 4)[0]
    torch.testing.assert_close(fields, permuted)
    empty = road_evidence([points[:0]], [0, 0, -1, 4, 4, 2], 4, 4)
    assert torch.isfinite(empty).all() and empty.count_nonzero() == 0


def test_continuous_paths_bridge_missing_rows_and_reject_isolated_false_peak():
    scores = torch.full((1, 1, 5, 7), -20.)
    for row, column in ((0, 1), (1, 2), (3, 4), (4, 5)):
        scores[0, 0, row, column] = 0
    scores[0, 0, 2] = 0  # no observation here
    scores[0, 0, 1, 6] = 3  # stronger local evidence, incompatible with the path
    scores.requires_grad_()
    probability = continuous_path_probabilities(scores, 1., max_step_m=1.)
    assert probability.argmax(-1).tolist() == [[[1, 2, 3, 4, 5]]]
    torch.testing.assert_close(probability.sum(-1), torch.ones(1, 1, 5))
    (probability * torch.arange(7.)).sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_cubic_fit_uses_observed_trajectory_and_extrapolates_known_bend():
    y = torch.arange(1., 100., 2.)
    values = 2 + .04 * y + .003 * y**2
    weights = (y < 70).float()
    weights[10:13] = 0
    corrupted = values.clone()
    corrupted[weights == 0] = -1000  # must never enter the fit
    controls = fit_observed_cubic(corrupted, weights, y, 100.)
    t = y / 100
    basis = torch.stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3), -1)
    torch.testing.assert_close(basis @ controls, values, atol=.03, rtol=0)
    empty = fit_observed_cubic(corrupted, torch.zeros_like(weights), y, 100.)
    assert torch.isfinite(empty).all() and empty.count_nonzero() == 0


def _model():
    return ContinuousBEVShapePrior(
        lidar_in_channels=4, hidden_channels=16, max_templates=5,
        pc_range=[-6, 0, -3, 6, 100, 3], longitudinal_rows=8,
        anchor_y_steps=[5., 20., 50., 100.])


def test_continuous_prior_preserves_cubic_budget_and_receives_training_gradients():
    torch.manual_seed(7)
    model = _model()
    road = torch.randn(2, 16, 16, 24) * .1
    road[:, 0] = 1
    inputs = dict(features=torch.randn(2, 4, 16, 24), road_evidence=road)
    prediction = model(inputs)
    assert prediction['shape_x_controls'].shape == (2, 5, 4)
    assert prediction['shape_z_controls'].shape == (2, 5, 4)
    assert prediction['shape_x_controls'][..., 0].count_nonzero() == 0
    assert prediction['shape_z_controls'][..., 0].count_nonzero() == 0
    lanes = prediction['shape_points'][:, :2].detach().clone()
    lanes[..., 0] += 2
    lanes[..., 2] += .01 * lanes[..., 1]
    losses = model.loss(prediction, lanes, torch.ones(2, 2, 4))
    assert 'loss_road_track_direction' in losses and 'loss_road_track_curvature' in losses
    total = sum(v for k, v in losses.items() if k.startswith('loss_'))
    total.backward()
    for parameter in (model.feature_encoder[0][0].weight, model.height_head.weight,
                      model.shape_tokens.weight, model.context_encoder[0][0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    expand = ShapeProposalExpander(
        pc_range=[-6, 0, -3, 6, 100, 3], anchor_y_steps=[5., 20., 50., 100.],
        num_x_seeds=101, template_feature_channels=16, embed_dims=16)
    proposals = expand(prediction)
    expected = prediction['shape_x_controls'][:, :, None] + expand.x_seeds[None, None, :, None]
    torch.testing.assert_close(proposals['curve_x_controls'], expected.reshape(2, 505, 4))


def test_empty_cloud_has_no_fabricated_curve_or_auxiliary_loss():
    model = _model()
    result = model(dict(features=torch.randn(1, 4, 16, 24),
                        road_evidence=torch.zeros(1, 16, 16, 24)))
    assert result['from_lidar'].tolist() == [False]
    assert result['shape_x_controls'].count_nonzero() == 0
    assert result['shape_z_controls'].count_nonzero() == 0
    lanes = torch.zeros(1, 2, 4, 3)
    lanes[:, :, :, 1] = model.anchor_y_steps
    losses = model.loss(result, lanes, torch.zeros(1, 2, 4))
    for key, value in losses.items():
        assert torch.isfinite(value)
        if key.startswith('loss_'):
            assert value == 0


def test_auxiliary_supervision_can_be_disabled_for_architecture_comparison():
    model = _model()
    model.direction_loss_weight = model.curvature_loss_weight = 0
    road = torch.zeros(1, 16, 16, 24)
    road[:, 0] = 1
    result = model(dict(features=torch.randn(1, 4, 16, 24), road_evidence=road))
    losses = model.loss(result, result['shape_points'][:, :1].detach(), torch.ones(1, 1, 4))
    assert not any(key.startswith('loss_road_track') for key in losses)
    assert any(key.startswith('loss_lidar_shape') for key in losses)
