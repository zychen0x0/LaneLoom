import pytest
import torch

from LaneFormer.models.head.lidar_shape_token_prior import (
    LaneFormerLiDARShapeTokenPrior,
)
from LaneFormer.models.head.shape_proposal_router import (
    LaneFormerShapeProposalRouter,
)


PC_RANGE = [-2., 0., -2., 2., 4., 2.]
ANCHORS = [0., 1., 2., 3., 4.]


def _prior(max_templates=3):
    return LaneFormerLiDARShapeTokenPrior(
        lidar_in_channels=8,
        pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS,
        max_templates=max_templates,
        hidden_channels=8,
        lateral_slope_scale=.1,
        vertical_slope_scale=.03,
        shape_nms_threshold=.5,
        shape_softmin_temperature=.2,
        shape_mode_loss_weight=1.0,
        shape_diversity_loss_weight=.05,
    )


def _metas(batch_size, image_size=8):
    lane2img = torch.tensor([
        [1.0, 0.0, 0.0, image_size / 2],
        [0.0, 1.0, 0.0, image_size / 2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    current = dict(
        lane2img=[lane2img], img_shape=(image_size, image_size))
    return [dict(current=current) for _ in range(batch_size)]


def test_shape_distance_ignores_global_x_and_z_translation():
    prior = _prior(max_templates=2)
    y = torch.tensor(ANCHORS)
    base = torch.stack((0.08 * y.square(), y, 0.05 * y), dim=-1)
    translated = base + torch.tensor([3., 0., 1.5])
    templates = base.view(1, 1, len(ANCHORS), 3)
    gt = translated.view(1, 1, len(ANCHORS), 3)
    visibility = torch.ones(1, 1, len(ANCHORS))

    distance, valid = prior.shape_distance(templates, gt, visibility)

    assert valid.tolist() == [[True]]
    assert distance.shape == (1, 1, 1)
    assert distance.item() < 1e-4


def test_invisible_nan_padding_cannot_poison_shape_distance():
    prior = _prior(max_templates=1)
    y = torch.tensor(ANCHORS)
    shape = torch.stack((0.08 * y.square(), y, 0.05 * y), dim=-1)
    gt = shape.clone()
    gt[-1] = torch.nan
    visibility = torch.tensor([[[1., 1., 1., 1., 0.]]])

    distance, valid = prior.shape_distance(
        shape.view(1, 1, len(ANCHORS), 3),
        gt.view(1, 1, len(ANCHORS), 3),
        visibility)

    assert valid.tolist() == [[True]]
    assert torch.isfinite(distance).all()
    assert distance.item() < 1e-4


def test_shape_prior_predicts_cubic_templates_and_deduplicates_without_confidence():
    prior = _prior(max_templates=3)
    output = prior([torch.randn(2, 8, 3, 3)])

    assert output['shape_templates'].shape == (2, 3, 4, 3)
    assert output['shape_points'].shape == (2, 3, len(ANCHORS), 3)
    assert output['shape_x_controls'].shape == (2, 3, 4)
    assert output['shape_z_controls'].shape == (2, 3, 4)
    assert torch.all(output['shape_x_controls'][..., 0] == 0)
    assert torch.all(output['shape_z_controls'][..., 0] == 0)
    assert output['shape_control_yaw_degrees'].shape == (2, 3, 3)
    assert output['shape_control_pitch_degrees'].shape == (2, 3, 3)
    assert output['shape_lateral_slopes'].shape == (2, 3, 3)
    assert output['shape_vertical_slopes'].shape == (2, 3, 3)
    assert 'confidence' not in output
    assert output['unique_template_indices'].shape == (2, 3)
    assert output['unique_template_mask'].shape == (2, 3)


def test_shape_prior_does_not_learn_gt_geometry_from_empty_lidar():
    prior = _prior(max_templates=2)
    output = prior(dict(
        features=torch.randn(1, 8, 3, 3),
        occupancy=torch.zeros(1, 1, 3, 3, dtype=torch.bool)))
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    visibility = torch.ones(1, 1, len(ANCHORS))

    losses = prior.loss(output, gt, visibility)

    assert output['from_lidar'].tolist() == [False]
    assert losses['loss_lidar_shape_cover'] == 0
    assert losses['loss_lidar_shape_mode'] == 0
    assert losses['loss_lidar_shape_diversity'] == 0
    assert losses['diag_unique_template_count'] == 0
    assert losses['diag_predicted_shape_metric_rank'] == 0
    assert losses['diag_gt_shape_metric_rank'] == 0


def test_shape_prior_converts_three_unbounded_slopes_to_controls():
    prior = _prior(max_templates=1)
    lateral_slopes = torch.tensor([0., .5, 2.])
    vertical_slopes = torch.tensor([.1, 0., -.2])
    raw = torch.cat((
        lateral_slopes / prior.lateral_slope_scale,
        vertical_slopes / prior.vertical_slope_scale))
    with torch.no_grad():
        prior.template_head[-1].weight.zero_()
        prior.template_head[-1].bias.copy_(raw)

    output = prior([torch.randn(1, 8, 3, 3)])
    dy = ANCHORS[-1] / 3
    expected_x = torch.cat((
        torch.zeros(1),
        torch.cumsum(dy * lateral_slopes, dim=0),
    ))
    expected_z = torch.cat((
        torch.zeros(1),
        torch.cumsum(dy * vertical_slopes, dim=0),
    ))

    assert torch.allclose(
        output['shape_lateral_slopes'][0, 0], lateral_slopes, atol=1e-5)
    assert torch.allclose(
        output['shape_vertical_slopes'][0, 0], vertical_slopes, atol=1e-5)
    assert torch.allclose(
        output['shape_control_yaw_degrees'][0, 0],
        torch.rad2deg(torch.atan(lateral_slopes)))
    assert torch.allclose(
        output['shape_control_pitch_degrees'][0, 0],
        torch.rad2deg(torch.atan(vertical_slopes)))
    assert torch.allclose(output['shape_x_controls'][0, 0], expected_x, atol=1e-5)
    assert torch.allclose(output['shape_z_controls'][0, 0], expected_z, atol=1e-5)


def test_unbounded_slopes_reconstruct_arbitrary_relative_cubic_controls():
    prior = _prior(max_templates=1)
    target_x = torch.tensor([0., 2., -3., 25.])
    target_z = torch.tensor([0., 1., -4., 7.])
    dy = ANCHORS[-1] / 3
    lateral_slopes = torch.diff(target_x) / dy
    vertical_slopes = torch.diff(target_z) / dy
    raw = torch.cat((
        lateral_slopes / prior.lateral_slope_scale,
        vertical_slopes / prior.vertical_slope_scale))
    with torch.no_grad():
        prior.template_head[-1].weight.zero_()
        prior.template_head[-1].bias.copy_(raw)

    output = prior([torch.randn(1, 8, 3, 3)])

    assert torch.allclose(output['shape_x_controls'][0, 0], target_x)
    assert torch.allclose(output['shape_z_controls'][0, 0], target_z)


@pytest.mark.parametrize('legacy_option', [
    'max_yaw_degrees',
    'compact_loss_weight',
])
def test_removed_shape_options_fail_instead_of_silently_changing_semantics(
        legacy_option):
    with pytest.raises(TypeError, match='were removed'):
        LaneFormerLiDARShapeTokenPrior(
            lidar_in_channels=8, pc_range=PC_RANGE,
            anchor_y_steps=ANCHORS, **{legacy_option: 30.})


def test_steep_unbounded_slopes_keep_shape_loss_and_gradients_finite():
    prior = _prior(max_templates=1)
    slopes = torch.tensor([5., -4., 3., 1., -.8, .5])
    scales = slopes.new_tensor([
        prior.lateral_slope_scale] * 3
        + [prior.vertical_slope_scale] * 3)
    with torch.no_grad():
        prior.template_head[-1].weight.zero_()
        prior.template_head[-1].bias.copy_(slopes / scales)
    output = prior([torch.randn(1, 8, 3, 3)])
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    visibility = torch.ones(1, 1, len(ANCHORS))

    losses = prior.loss(output, gt, visibility)
    total = sum(losses.values())
    total.backward()

    assert torch.isfinite(output['shape_points']).all()
    assert torch.isfinite(total)
    assert torch.isfinite(prior.template_head[-1].bias.grad).all()


def test_true_softmin_gives_every_template_a_distance_reducing_gradient():
    prior = _prior(max_templates=2)
    distance = torch.tensor([[[0.1, 1.0]]], requires_grad=True)

    prior._softmin(distance, temperature=.25).sum().backward()

    assert torch.all(distance.grad > 0)


def test_shape_nms_keeps_one_medoid_per_distinct_shape_cluster():
    prior = _prior(max_templates=3)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    turn = torch.stack((0.15 * y.square(), y, torch.zeros_like(y)), -1)
    shape_points = torch.stack((straight, straight, turn), dim=0).unsqueeze(0)

    unique, valid = prior._unique_template_indices(shape_points)

    assert valid[0].sum() == 2
    assert 2 in unique[0, valid[0]].tolist()


def test_complete_link_clustering_does_not_merge_a_similarity_chain():
    prior = _prior(max_templates=3)
    pairwise = torch.tensor([[
        [0.0, 0.4, 0.8],
        [0.4, 0.0, 0.4],
        [0.8, 0.4, 0.0],
    ]])

    unique, valid = prior._complete_link_medoids(pairwise, threshold=.5)

    assert valid[0].sum() == 2
    assert unique[0, valid[0]].unique().numel() == 2


def test_shape_coverage_allows_one_template_to_cover_parallel_gt_lanes():
    prior = _prior(max_templates=2)
    y = torch.tensor(ANCHORS)
    shape = torch.stack((0.05 * y.square(), y, 0.03 * y), dim=-1)
    templates = torch.stack((shape, shape), dim=0).unsqueeze(0)
    lanes = torch.stack((
        shape + torch.tensor([-1., 0., .2]),
        shape + torch.tensor([1., 0., -.4]),
    ), dim=0).unsqueeze(0)
    visibility = torch.ones(1, 2, len(ANCHORS))
    prior_output = dict(
        shape_points=templates,
        from_lidar=True,
    )

    losses = prior.loss(prior_output, lanes, visibility)

    assert set(losses) == {
        'loss_lidar_shape_cover', 'loss_lidar_shape_mode',
        'loss_lidar_shape_diversity', 'diag_unique_template_count',
        'diag_predicted_shape_metric_rank',
        'diag_gt_shape_metric_rank', 'diag_shape_lane_cover_recall',
        'diag_shape_mode_cover_recall', 'diag_shape_mode_overflow_rate',
        'diag_matched_template_min_distance'}
    assert losses['loss_lidar_shape_cover'].item() < 1e-4
    assert losses['loss_lidar_shape_mode'].item() < 1e-4
    assert losses['loss_lidar_shape_diversity'] == 0
    assert losses['diag_gt_shape_metric_rank'] == 1
    assert losses['diag_shape_lane_cover_recall'] == 1
    assert losses['diag_unique_template_count'] == 1
    assert losses['diag_predicted_shape_metric_rank'] == 1


def test_shape_similarity_requires_geometry_heading_and_grade_thresholds():
    prior = _prior(max_templates=2)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    translated = straight + torch.tensor([2., 0., .5])
    turn = torch.stack((.10 * y.square(), y, torch.zeros_like(y)), -1)
    templates = torch.stack((straight, turn), dim=0).unsqueeze(0)
    lanes = translated.view(1, 1, len(ANCHORS), 3)
    visibility = torch.ones(1, 1, len(ANCHORS))

    _, valid, components = prior.shape_distance(
        templates, lanes, visibility, return_components=True)
    similar = prior._shape_similar(components)

    assert valid.tolist() == [[True]]
    assert similar.tolist() == [[[True, False]]]
    assert components['aligned_p80'][0, 0, 0] < 1e-5


def test_gt_metric_rank_clusters_parallel_lanes_but_separates_turns():
    prior = _prior(max_templates=3)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    parallel = straight + torch.tensor([1.5, 0., .2])
    turn = torch.stack((.12 * y.square(), y, torch.zeros_like(y)), -1)
    lanes = torch.stack((straight, parallel, turn), dim=0).unsqueeze(0)
    visibility = torch.ones(1, 3, len(ANCHORS))

    medoids, active, _ = prior._gt_shape_modes(lanes, visibility)

    assert active.sum() == 2
    assert medoids[0, active[0]].unique().numel() == 2


def test_two_gt_modes_require_two_distinct_templates():
    prior = _prior(max_templates=2)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    turn = torch.stack((.12 * y.square(), y, torch.zeros_like(y)), -1)
    lanes = torch.stack((straight, turn), dim=0).unsqueeze(0)
    visibility = torch.ones(1, 2, len(ANCHORS))
    collapsed = torch.stack((straight, straight), dim=0).unsqueeze(0)
    covered = torch.stack((straight, turn), dim=0).unsqueeze(0)

    collapsed_losses = prior.loss(
        dict(shape_points=collapsed, from_lidar=True), lanes, visibility)
    covered_losses = prior.loss(
        dict(shape_points=covered, from_lidar=True), lanes, visibility)

    assert collapsed_losses['diag_gt_shape_metric_rank'] == 2
    assert collapsed_losses['loss_lidar_shape_mode'] > (
        covered_losses['loss_lidar_shape_mode'] + .1)
    assert collapsed_losses['loss_lidar_shape_diversity'] > 0
    assert covered_losses['loss_lidar_shape_diversity'] == 0
    assert covered_losses['diag_shape_mode_cover_recall'] == 1


def test_five_template_mode_assignment_is_exact_and_stays_on_device():
    prior = _prior(max_templates=3)
    cost = torch.tensor([[
        [1., 2., 100.],
        [1., 100., 100.],
        [100., 1., 1.],
    ]])
    active = torch.ones(1, 3, dtype=torch.bool)

    assignments = prior._assign_modes(cost, active)

    assert assignments.device == cost.device
    assert assignments.tolist() == [[1, 0, 2]]


def test_one_gt_mode_does_not_force_all_five_templates_to_rank_five():
    prior = _prior(max_templates=5)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    templates = straight.view(1, 1, len(ANCHORS), 3).expand(1, 5, -1, -1)
    lanes = straight.view(1, 1, len(ANCHORS), 3)
    visibility = torch.ones(1, 1, len(ANCHORS))

    losses = prior.loss(
        dict(shape_points=templates, from_lidar=True), lanes, visibility)

    assert losses['diag_gt_shape_metric_rank'] == 1
    assert losses['loss_lidar_shape_cover'].item() < 1e-4
    assert losses['loss_lidar_shape_mode'].item() < 1e-4
    assert losses['loss_lidar_shape_diversity'] == 0


def test_short_gt_fragment_does_not_create_a_new_shape_mode():
    prior = _prior(max_templates=2)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    short_turn = torch.stack((.2 * y.square(), y, torch.zeros_like(y)), -1)
    lanes = torch.stack((straight, short_turn), dim=0).unsqueeze(0)
    visibility = torch.tensor([[
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1.],
    ]])

    _, active, _ = prior._gt_shape_modes(lanes, visibility)

    assert active.sum() == 1


def test_gt_modes_require_four_shared_y_anchors_to_merge():
    prior = _prior(max_templates=2)
    y = torch.tensor(ANCHORS)
    straight = torch.stack((torch.zeros_like(y), y, torch.zeros_like(y)), -1)
    translated = straight + torch.tensor([2., 0., .3])
    lanes = torch.stack((straight, translated), dim=0).unsqueeze(0)
    visibility = torch.tensor([[
        [1., 1., 1., 1., 0.],
        [0., 1., 1., 1., 1.],
    ]])

    _, active, _ = prior._gt_shape_modes(lanes, visibility)

    assert active.sum() == 2


def test_proposal_router_expands_unique_shapes_and_returns_fixed_budget():
    prior = _prior(max_templates=3)
    prior_output = prior([torch.randn(2, 8, 3, 3)])
    # Exercise variable K explicitly: sample 0 uses one shape, sample 1 two.
    prior_output['unique_template_indices'] = torch.tensor([
        [0, 0, 0], [0, 1, 0]])
    prior_output['unique_template_mask'] = torch.tensor([
        [True, False, False], [True, True, False]])
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8,
        embed_dims=8,
        pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS,
        num_x_seeds=4,
        proposal_budget=4,
        hidden_channels=8,
    )
    image_features = [torch.randn(2, 3, 1, 8, 8, 8)]

    routed = router(prior_output, image_features, _metas(2))

    assert routed['curve_x_controls'].shape == (2, 4, 4)
    assert routed['curve_z_controls'].shape == (2, 4, 4)
    assert routed['anchor_points'].shape == (2, 4, len(ANCHORS), 3)
    assert routed['query_features'].shape == (2, 4, 8)
    assert routed['routing_scores'].shape == (2, 4)
    assert routed['candidate_points'].shape[:2] == (2, 12)
    assert routed['candidate_valid_mask'][0].sum() == 4
    assert routed['candidate_valid_mask'][1].sum() == 8


def test_router_loss_supervises_absolute_shape_seed_placement():
    prior = _prior(max_templates=2)
    prior_output = prior([torch.randn(1, 8, 3, 3)])
    prior_output['unique_template_indices'] = torch.tensor([[0, 0]])
    prior_output['unique_template_mask'] = torch.tensor([[True, False]])
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8,
        embed_dims=8,
        pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS,
        num_x_seeds=4,
        proposal_budget=4,
        hidden_channels=8,
        q0_distance_threshold=.3,
        q0_match_ratio=.8,
    )
    routed = router(
        prior_output, [torch.randn(1, 3, 1, 8, 8, 8)], _metas(1))
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[0, 0, :, 1] = torch.tensor(ANCHORS)
    gt[0, 0, :, 0] = -.5
    visibility = torch.ones(1, 1, len(ANCHORS))

    losses = router.loss(routed, gt, visibility)
    sum(losses.values()).backward()

    assert set(losses) == {'loss_proposal_router', 'loss_q0_geometry'}
    assert torch.isfinite(losses['loss_proposal_router'])
    assert torch.isfinite(losses['loss_q0_geometry'])
    assert router.score_head[-1].weight.grad is not None
    assert torch.isfinite(router.score_head[-1].weight.grad).all()
    assert prior.template_head[-1].weight.grad is not None
    assert torch.isfinite(prior.template_head[-1].weight.grad).all()


def test_router_q0_standard_is_strictly_tighter_than_evaluation():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=2, proposal_budget=2,
        hidden_channels=8, q0_distance_threshold=.3, q0_match_ratio=.8)
    candidates = torch.zeros(1, 2, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[0, 0, :, 0] = torch.tensor([.2, .2, .2, .2, .8])
    candidates[0, 1, :, 0] = torch.tensor([.2, .2, .2, .31, .31])
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    visibility = torch.ones(1, 1, len(ANCHORS))

    strict, _, _ = router._strict_candidate_matches(
        candidates, torch.ones(1, 2, dtype=torch.bool), gt, visibility)

    assert strict[0, 0, 0]
    assert not strict[0, 1, 0]


def test_q0_geometry_loss_only_requires_one_strict_candidate_per_gt():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=2, proposal_budget=2,
        hidden_channels=8, q0_distance_threshold=.3, q0_match_ratio=.8)
    candidates = torch.zeros(1, 2, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[0, 1, :, 0] = 3.
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    visibility = torch.ones(1, 1, len(ANCHORS))

    loss = router._q0_geometry_loss(
        candidates, torch.ones(1, 2, dtype=torch.bool), gt, visibility)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_router_q0_loss_checks_selected_decoder_queries():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=2, proposal_budget=1,
        hidden_channels=8, q0_distance_threshold=.3, q0_match_ratio=.8)
    candidate_points = torch.zeros(1, 2, len(ANCHORS), 3)
    candidate_points[..., 1] = torch.tensor(ANCHORS)
    selected_points = candidate_points[:, :1].clone()
    selected_points[..., 0] = 1.
    routed = dict(
        candidate_points=candidate_points,
        candidate_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        candidate_routing_logits=torch.zeros(1, 2),
        anchor_points=selected_points,
        routing_scores=torch.full((1, 1), .5))
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    visibility = torch.ones(1, 1, len(ANCHORS))

    losses = router.loss(routed, gt, visibility)

    assert losses['loss_q0_geometry'] > 0


def test_router_uses_one_positive_candidate_per_gt():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=3, proposal_budget=3,
        hidden_channels=8, q0_distance_threshold=.3, q0_match_ratio=.8)
    candidates = torch.zeros(1, 3, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[:, 1, :, 0] = .1
    candidates[:, 2, :, 0] = .2
    gt = torch.zeros(1, 1, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    visibility = torch.ones(1, 1, len(ANCHORS))

    targets = router._routing_targets(
        candidates, torch.ones(1, 3, dtype=torch.bool), gt, visibility)

    assert targets.sum() == 1


def test_router_assignment_globally_maximizes_strict_matches():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=2, proposal_budget=2,
        hidden_channels=8, q0_distance_threshold=.3, q0_match_ratio=.8)
    candidates = torch.zeros(1, 2, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[0, 0, :, 0] = .1
    candidates[0, 1, :, 0] = -.2
    gt = torch.zeros(1, 2, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    gt[0, 1, :, 0] = .2
    visibility = torch.ones(1, 2, len(ANCHORS))

    assignments = router._routing_assignments(
        candidates, torch.ones(1, 2, dtype=torch.bool), gt, visibility)

    assert assignments.tolist() == [[1, 0]]


def test_q0_geometry_loss_requires_exclusive_gt_coverage():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=2, proposal_budget=2,
        hidden_channels=8, q0_distance_threshold=.3, q0_match_ratio=.8)
    candidates = torch.zeros(1, 2, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[0, 1, :, 0] = 2.
    gt = torch.zeros(1, 2, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    gt[0, 1, :, 0] = .1
    visibility = torch.ones(1, 2, len(ANCHORS))

    loss = router._q0_geometry_loss(
        candidates, torch.ones(1, 2, dtype=torch.bool), gt, visibility)

    assert loss > 0


def test_router_preserves_exact_shape_translation_outside_bev_boundaries():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=4, proposal_budget=4,
        hidden_channels=8)
    shape_x = torch.tensor([[[0., 1., 2., 3.]]])
    prior_output = dict(
        shape_x_controls=shape_x,
        shape_z_controls=torch.zeros_like(shape_x),
        unique_template_indices=torch.tensor([[0]]),
        unique_template_mask=torch.tensor([[True]]),
    )

    candidate_x, _, valid, _, _ = router._expand_candidates(prior_output)

    expected = shape_x[0, 0] + router.x_seeds[-1]
    assert valid[0].sum() == 4
    assert torch.allclose(candidate_x[0, 3], expected)
    assert candidate_x[0, 3, -1] > PC_RANGE[3]


def test_router_guarantees_nearest_candidate_for_every_gt_is_positive():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=4, proposal_budget=4,
        hidden_channels=8, q0_distance_threshold=.01,
        q0_match_ratio=1.0)
    candidates = torch.zeros(1, 4, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[:, :, :, 0] = torch.arange(4).view(1, 4, 1).float()
    gt = torch.zeros(1, 2, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    gt[0, 0, :, 0] = 10.
    gt[0, 1, :, 0] = -10.
    visibility = torch.ones(1, 2, len(ANCHORS))
    candidate_valid = torch.ones(1, 4, dtype=torch.bool)

    targets = router._routing_targets(
        candidates, candidate_valid, gt, visibility)

    assert targets.sum() == 2
    assert targets[0, 0]
    assert targets[0, 3]


def test_router_guarantees_distinct_candidates_when_gt_nearest_collides():
    router = LaneFormerShapeProposalRouter(
        image_in_channels=8, embed_dims=8, pc_range=PC_RANGE,
        anchor_y_steps=ANCHORS, num_x_seeds=4, proposal_budget=4,
        hidden_channels=8, q0_distance_threshold=.01,
        q0_match_ratio=1.0)
    candidates = torch.zeros(1, 4, len(ANCHORS), 3)
    candidates[..., 1] = torch.tensor(ANCHORS)
    candidates[:, :, :, 0] = torch.arange(4).view(1, 4, 1).float()
    gt = torch.zeros(1, 2, len(ANCHORS), 3)
    gt[..., 1] = torch.tensor(ANCHORS)
    gt[0, 0, :, 0] = 10.
    gt[0, 1, :, 0] = 9.
    visibility = torch.ones(1, 2, len(ANCHORS))

    targets = router._routing_targets(
        candidates, torch.ones(1, 4, dtype=torch.bool), gt, visibility)

    assert targets.sum() == 2
    assert targets[0, 2]
    assert targets[0, 3]
