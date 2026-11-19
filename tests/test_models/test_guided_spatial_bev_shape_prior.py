"""Predicted lane fields guide Q0 without annotations in its forward pass."""
import torch
import torch.nn.functional as F
import pytest

from LaneLoom.models.head.guided_spatial_bev_shape_prior import GuidedSpatialBEVShapePrior
from LaneLoom.models.head.shape_proposal_expander import ShapeProposalExpander
from LaneLoom.models.losses.bev_lane_field import bev_lane_field_loss


RANGE = [-25, 0, -5, 25, 100, 5]


def test_measured_geometry_changes_prediction_and_receives_gradients_without_gt():
    torch.manual_seed(19)
    model = GuidedSpatialBEVShapePrior(
        lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], max_templates=5,
        longitudinal_rows=8, spatial_layers=1, num_heads=4,
        geometric_readout=True, use_road_evidence=True).eval()
    road = torch.zeros(1, 16, 16, 24)
    road[:, 0, :, 8:16] = 1
    road[:, 2:4, :, 8:16] = .3
    road[:, 6:9, :, 8:16] = .5
    road.requires_grad_()
    inputs = dict(features=torch.randn(1, 4, 16, 24),
                  occupancy=torch.ones(1, 1, 16, 24, dtype=torch.bool),
                  road_evidence=road)
    output = model(inputs)
    assert output['shape_x_controls'].shape == (1, 5, 4)
    assert torch.isfinite(output['shape_points']).all()
    (output['shape_points'].square().mean() + output['lane_field_logits'].square().mean()).backward()
    # The neutral bypass gets a learning signal on the first backward pass.
    weight = model.feature_encoder[0].weight
    assert weight.grad[:, 4:].abs().sum() > 0
    with torch.no_grad():
        weight[:, 4:] -= .001 * weight.grad[:, 4:]
    model.zero_grad(set_to_none=True)
    road.grad = None
    output = model(inputs)
    (output['shape_points'].square().mean() + output['lane_field_logits'].square().mean()).backward()
    assert road.grad[:, 2:4, :, 8:16].abs().sum() > 0
    assert road.grad[:, 6:9, :, 8:16].abs().sum() > 0
    assert torch.count_nonzero(road.grad[..., :8]) == 0
    changed = road.detach().clone()
    changed[:, 2:4, :, 8:16] += .7
    with torch.no_grad():
        moved = model(dict(inputs, road_evidence=changed))
        assert not torch.allclose(output['shape_points'], moved['shape_points'])
        # Missing returns may contain arbitrary placeholders, never evidence.
        changed.copy_(road.detach())
        changed[:, 1:, :, :8] = 100
        unknown = model(dict(inputs, road_evidence=changed))
        torch.testing.assert_close(output['shape_points'], unknown['shape_points'], rtol=0, atol=0)


def test_geometry_extension_preserves_shared_initialization_and_initial_function():
    arguments = dict(lidar_in_channels=4, hidden_channels=16,
                     anchor_y_steps=[5., 20., 50., 100.],
                     longitudinal_rows=8, spatial_layers=1, num_heads=4,
                     geometric_readout=True)
    torch.manual_seed(31)
    old = GuidedSpatialBEVShapePrior(**arguments).eval()
    torch.manual_seed(31)
    extended = GuidedSpatialBEVShapePrior(**arguments, use_road_evidence=True).eval()
    for key, value in old.state_dict().items():
        changed = extended.state_dict()[key]
        if key == 'feature_encoder.0.weight':
            torch.testing.assert_close(value, changed[:, :4], rtol=0, atol=0)
            assert torch.count_nonzero(changed[:, 4:]) == 0
        else:
            torch.testing.assert_close(value, changed, rtol=0, atol=0)
    inputs = dict(features=torch.randn(1, 4, 16, 24),
                  occupancy=torch.ones(1, 1, 16, 24, dtype=torch.bool),
                  road_evidence=torch.randn(1, 16, 16, 24))
    inputs['road_evidence'][:, 0] = 1
    with torch.no_grad():
        before, after = old(inputs), extended(inputs)
    # A larger convolution can use a different reduction kernel on GPU.
    torch.testing.assert_close(before['shape_points'], after['shape_points'], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(before['lane_field_logits'], after['lane_field_logits'], rtol=1e-5, atol=1e-5)


def test_measured_geometry_requires_native_grid():
    model = GuidedSpatialBEVShapePrior(
        lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], longitudinal_rows=8,
        spatial_layers=1, num_heads=4, use_road_evidence=True)
    features = torch.randn(1, 4, 16, 24)
    for road in (None, torch.zeros(1, 16, 8, 12), torch.zeros(1, 15, 16, 24)):
        with pytest.raises(ValueError, match='Road evidence'):
            model(dict(features=features, road_evidence=road))


def make_prior():
    return GuidedSpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
                                     anchor_y_steps=[5., 20., 50., 100.],
                                     max_templates=5, longitudinal_rows=8,
                                     spatial_layers=1, num_heads=4)


def test_guided_prior_keeps_seed_budget_and_backpropagates_field_supervision():
    torch.manual_seed(7)
    model = make_prior()
    inputs = dict(features=torch.randn(2, 4, 16, 24),
                  occupancy=torch.ones(2, 1, 16, 24, dtype=torch.bool))
    output = model(inputs)
    assert output['lane_field_logits'].shape == (2, 1, 16, 24)
    assert output['shape_x_controls'].shape == (2, 5, 4)
    points = torch.tensor([[[[-2., 20., .3], [-1., 30., .4]],
                            [[2., 20., .3], [3., 30., .4]]]]).expand(2, -1, -1, -1)
    losses = bev_lane_field_loss(output, points, torch.ones(2, 2, 2, dtype=torch.bool), RANGE)
    (sum(v for k, v in losses.items() if k.startswith('loss_'))
     + output['shape_points'].square().mean()).backward()
    assert model.lane_field_head[-1].weight.grad.abs().sum() > 0
    assert model.feature_encoder[0].weight.grad.abs().sum() > 0
    expander = ShapeProposalExpander(pc_range=RANGE, anchor_y_steps=[5., 20., 50., 100.],
                                     num_x_seeds=101, template_feature_channels=16, embed_dims=16)
    expanded = expander(output)
    assert expanded['curve_x_controls'].shape == (2, 505, 4)
    expected = output['shape_x_controls'][:, :, None] + expander.x_seeds[None, None, :, None]
    torch.testing.assert_close(expanded['curve_x_controls'], expected.reshape(2, 505, 4))


def test_several_lanes_share_a_row_without_competing_and_unknown_rows_are_ignored():
    logits = torch.zeros(1, 1, 10, 100, requires_grad=True)
    height = torch.zeros_like(logits, requires_grad=True)
    # Seven lanes: the dense target is not limited to five template instances.
    points = torch.tensor([[[[x, 25., .4]] for x in [-9.75, -6.75, -3.75, .25, 3.25, 6.25, 9.25]]])
    losses = bev_lane_field_loss(dict(lane_field_logits=logits, lane_height_field=height),
                                  points, torch.ones(1, 7, 1, dtype=torch.bool), RANGE)
    (losses['loss_bev_lane_position'] + losses['loss_bev_lane_height']).backward()
    assert losses['diag_bev_supervised_rows'] == 1
    for x in points[0, :, 0, 0]:
        column = int((x + 25) * 2)
        assert logits.grad[0, 0, 2, column] < 0
    assert logits.grad[0, 0, 2, 0] > 0
    assert torch.count_nonzero(logits.grad[0, 0, [0, 1, 3, 4, 5, 6, 7, 8, 9]]) == 0
    assert height.grad.abs().sum() > 0


def test_empty_annotations_or_cloud_do_not_create_background_or_nan_losses():
    for cloud_present in [True, False]:
        logits = torch.randn(1, 1, 8, 16, requires_grad=True)
        height = torch.randn(1, 1, 8, 16, requires_grad=True)
        points = torch.full((1, 1, 2, 3), float('nan')) if cloud_present else torch.zeros(1, 1, 2, 3)
        visible = torch.zeros(1, 1, 2, dtype=torch.bool) if cloud_present else torch.ones(1, 1, 2, dtype=torch.bool)
        losses = bev_lane_field_loss(dict(lane_field_logits=logits, lane_height_field=height,
                                          from_lidar=torch.tensor([cloud_present])), points, visible, RANGE)
        total = losses['loss_bev_lane_position'] + losses['loss_bev_lane_height']
        assert total == 0
        total.backward()
        assert torch.count_nonzero(logits.grad) == 0
        assert torch.count_nonzero(height.grad) == 0


def test_predicted_field_changes_q0_at_inference_and_empty_clouds_stay_finite():
    torch.manual_seed(11)
    model = make_prior().eval()
    inputs = dict(features=torch.randn(1, 4, 16, 24),
                  occupancy=torch.zeros(1, 1, 16, 24, dtype=torch.bool))
    with torch.no_grad():
        empty = model(inputs)
        assert not empty['from_lidar'].item()
        assert torch.isfinite(empty['shape_points']).all()
        assert torch.count_nonzero(empty['shape_x_controls']) == 0
        # Lane locations can be between actual returns; sparse occupancy must
        # not block all spatial field predictions from contributing to Q0.
        inputs['occupancy'][..., 4, 3] = True
        first = model(inputs)['shape_points']
        model.lane_field_head[-1].bias[1] += 1.
        second = model(inputs)['shape_points']
    assert (first-second).abs().max() > 1e-5


def test_geometric_readout_recovers_analytic_road_and_keeps_absolute_height():
    model = GuidedSpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], max_templates=5, longitudinal_rows=8,
        spatial_layers=1, num_heads=4, geometric_readout=True,
        pc_range=[-8, 0, -5, 8, 100, 5])
    y = (torch.arange(8)+.5)*100/8
    attention = torch.zeros(1, 8, 5, 17)
    for row in range(8):
        attention[:, row, :, row+4] = 1.
    attention.requires_grad_()
    # The absolute path is x(y) = -4 + .08*y, z(y) = .4 + .01*y.
    height = (.4+.01*y)[None, :, None].expand(1, 8, 16).clone().requires_grad_()
    controls, _ = model._geometric_controls(torch.zeros(5, 8, 16), attention,
        torch.zeros(1, 8, 16), height, torch.ones(1, 8, 16, dtype=torch.bool))
    result = model._format_prior(controls)
    torch.testing.assert_close(result['shape_points'][0, 0, :, 0],
                               .08*model.anchor_y_steps, atol=.002, rtol=0)
    torch.testing.assert_close(result['shape_points'][0, 0, :, 2],
                               .4+.01*model.anchor_y_steps, atol=.002, rtol=0)
    assert abs(result['shape_z_controls'][0, 0, 0]-.4) < .002
    result['shape_points'].square().mean().backward()
    assert height.grad.abs().sum() > 0 and torch.isfinite(height.grad).all()
    assert attention.grad.abs().sum() > 0 and torch.isfinite(attention.grad).all()


def test_geometric_readout_handles_empty_rows_and_trains_field_through_q0_alone():
    from LaneLoom.models.losses.q0_visible_geometry import raw_visible_geometry_loss
    model = GuidedSpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], max_templates=5, longitudinal_rows=8,
        spatial_layers=1, num_heads=4, geometric_readout=True)
    occupied = torch.zeros(2, 1, 16, 24, dtype=torch.bool)
    occupied[0, :, :8] = True
    output = model(dict(features=torch.randn(2, 4, 16, 24), occupancy=occupied))
    assert torch.count_nonzero(output['shape_z_controls'][1]) == 0
    assert torch.isfinite(output['shape_points']).all()
    expander = ShapeProposalExpander(pc_range=RANGE, anchor_y_steps=[5., 20., 50., 100.],
                                     num_x_seeds=101, template_feature_channels=16, embed_dims=16)
    expanded = expander(output)
    points = torch.tensor([[[[0., 10., .5], [.2, 20., .7]]]]).expand(2, -1, -1, -1)
    losses = raw_visible_geometry_loss(expanded, points, torch.ones(2, 1, 2, dtype=torch.bool))
    losses['loss_q0_raw_geometry'].backward()
    assert model.lane_field_head[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(model.lane_field_head[-1].weight.grad).all()


def test_row_distribution_keeps_multiple_peaks_without_broadening_the_soft_target():
    width = 250
    column_x = -25+(torch.arange(width)+.5)*50/width
    points = torch.tensor([[[[-3.9, 25., 0.]], [[3.9, 25., 0.]]]])
    target = torch.exp(-.5*((column_x[:, None]-points[0, :, 0, 0])/.3).square()).amax(-1)
    q = target/target.sum()
    optimum_logits = q.clamp_min(1e-30).log()[None, None, None].expand(1, 1, 10, -1).clone().requires_grad_()
    output = dict(lane_field_logits=optimum_logits, lane_height_field=torch.zeros_like(optimum_logits),
                  lane_field_mode='row_distribution')
    result = bev_lane_field_loss(output, points, torch.ones(1, 2, 1, dtype=torch.bool), RANGE)
    assert abs(result['loss_bev_lane_position']) < 1e-6
    result['loss_bev_lane_position'].backward()
    assert optimum_logits.grad.abs().max() < 1e-6
    # Both lanes have probability peaks; a broad plateau is penalized despite
    # having the same high sigmoid score at the two annotated locations.
    plateau = torch.full_like(optimum_logits, -10.)
    plateau[..., (column_x >= -5) & (column_x <= 5)] = 10.
    broad = bev_lane_field_loss(dict(output, lane_field_logits=plateau), points,
                                torch.ones(1, 2, 1, dtype=torch.bool), RANGE)
    assert broad['loss_bev_lane_position'] > 1
    p = optimum_logits.detach().softmax(-1)[0, 0, 2]
    assert abs(p[105]-p[144]) < 1e-6
    assert p[125] < p[105]*.01


def test_row_distribution_forward_has_finite_geometry():
    model = GuidedSpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], max_templates=5, longitudinal_rows=8,
        spatial_layers=1, num_heads=4, geometric_readout=True, lane_field_mode='row_distribution')
    occupancy = torch.zeros(2, 1, 16, 24, dtype=torch.bool)
    occupancy[0, :, :8] = True
    output = model(dict(features=torch.randn(2, 4, 16, 24), occupancy=occupancy))
    assert output['lane_field_mode'] == 'row_distribution'
    assert torch.isfinite(output['shape_points']).all()
    assert torch.count_nonzero(output['shape_z_controls'][1]) == 0
    output['shape_points'].square().mean().backward()
    assert torch.isfinite(model.lane_field_head[-1].weight.grad).all()


def test_predicted_row_references_preserve_two_different_bends_instead_of_averaging_them():
    from LaneLoom.tools.evaluate_q0_prior import evaluate_controls
    import numpy as np
    model = GuidedSpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], max_templates=5, longitudinal_rows=50,
        spatial_layers=1, num_heads=4, geometric_readout=True,
        lane_field_mode='row_distribution', row_query_references=True)
    y = (torch.arange(50)+.5)*2
    x = -25+(torch.arange(250)+.5)*.2
    left, right = -5-.001*y*y, 5+.001*y*y
    scores = torch.stack((-.5*((x[None]-left[:, None])/.3).square(),
                           -.5*((x[None]-right[:, None])/.3).square()), -1).logsumexp(-1)[None]
    height = (.2+.01*y)[None, :, None].expand(1, 50, 250)
    observed = torch.ones(1, 50, 250, dtype=torch.bool)
    evidence = torch.zeros(5, 50, 16)
    # With uninformative content scores, global expectation averages two
    # genuinely different bends into a straight line. Reference-conditioned
    # reads preserve both bends using the same five-template/query budget.
    local = model._position_attention_bias(scores).softmax(-1)
    global_read = scores.softmax(-1)[:, :, None].expand(-1, -1, 5, -1)
    predictions = []
    for probability in [global_read, local]:
        controls, _ = model._geometric_controls(evidence, F.pad(probability, (0, 1)), scores, height, observed)
        predictions.append(model._format_prior(controls))
    raw_y = np.linspace(5, 95, 40)
    lanes = [dict(json_index=0, points=np.stack((-5-.001*raw_y**2, raw_y, .2+.01*raw_y), -1)),
             dict(json_index=1, points=np.stack((5+.001*raw_y**2, raw_y, .2+.01*raw_y), -1))]
    assert evaluate_controls(predictions[0], lanes)['paired_xz_p80']['covered'] == 0
    assert evaluate_controls(predictions[1], lanes)['paired_xz_p80']['covered'] == 2


def test_reference_conditioned_attention_is_finite_for_sparse_and_empty_clouds():
    model = GuidedSpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
        anchor_y_steps=[5., 20., 50., 100.], max_templates=5, longitudinal_rows=8,
        spatial_layers=1, num_heads=4, geometric_readout=True,
        lane_field_mode='row_distribution', row_query_references=True)
    occupancy = torch.zeros(2, 1, 16, 24, dtype=torch.bool)
    occupancy[0, :, :8] = True
    result = model(dict(features=torch.randn(2, 4, 16, 24), occupancy=occupancy))
    assert torch.isfinite(result['shape_points']).all()
    assert result['from_lidar'].tolist() == [True, False]
    assert torch.count_nonzero(result['shape_z_controls'][1]) == 0
    result['shape_points'].square().mean().backward()
    assert torch.isfinite(model.lane_field_head[-1].weight.grad).all()
