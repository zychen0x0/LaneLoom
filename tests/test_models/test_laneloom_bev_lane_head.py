from unittest.mock import Mock, patch

import pytest
import torch

from LaneLoom.models.head.laneloom_bev_lane_head import BEVLaneHead
from LaneLoom.models.head.laneloom_lane_query_decoder import (
    LaneQueryDecoder,
)
from LaneLoom.models.losses.laneloom_topk_loss import (
    LaneLoss,
    TopkLaneLoss,
)
from LaneLoom.models.losses.hungarian_assigner import (
    HungarianAssigner,
)
from LaneLoom.models.losses.topk_assigner import TopkAssigner


def _metas(batch_size, image_size=16):
    lidar2img = torch.tensor([
        [1.0, 0.0, 0.0, image_size / 2],
        [0.0, 1.0, 0.0, image_size / 2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    frame = dict(
        lidar2img=[lidar2img], img_shape=(image_size, image_size),
        lidaradj2lidarcurr=torch.eye(4))
    return [dict(queue={index: dict(frame) for index in range(3)},
                 current=dict(frame)) for _ in range(batch_size)]


def _head(embed_dims, bev_h, bev_w, num_query, anchors, lidar_channels,
          dn_num_groups=0, max_templates=1):
    if num_query % max_templates:
        raise ValueError('num_query must be divisible by max_templates.')
    return BEVLaneHead(
        anchor_prior=dict(
            type='LiDARShapeTokenPrior',
            lidar_in_channels=lidar_channels,
            anchor_y_steps=anchors,
            max_templates=max_templates,
            hidden_channels=8),
        proposal_expander=dict(
            type='ShapeProposalExpander',
            embed_dims=embed_dims,
            pc_range=[-25., 0., -5., 25., 100., 5.],
            anchor_y_steps=anchors,
            num_x_seeds=num_query // max_templates,
            template_feature_channels=8),
        bev_encoder=dict(
            type='BEVEncoder', bev_h=bev_h, bev_w=bev_w,
            lidar_bev_adapter=dict(
                type='LiDARBEVAdapter', embed_dims=embed_dims,
                lidar_in_channels=lidar_channels)),
        query_sampler=dict(
            type='DualPlaneTemporalSampler',
            image_in_channels=embed_dims, lidar_in_channels=embed_dims,
            embed_dims=embed_dims,
            pc_range=[-25., 0., -5., 25., 100., 5.],
            num_image_levels=1, temporal_kernel_size=3),
        lane_decoder=dict(
            type='LaneQueryDecoder', embed_dims=embed_dims,
            pc_range=[-25., 0., -5., 25., 100., 5.],
            num_query=num_query, anchor_y_steps=anchors, num_layers=2,
            num_heads=4, feedforward_channels=embed_dims * 2, dropout=0.0,
            dn_num_groups=dn_num_groups),
        loss_module=dict(
            type='TopkLaneLoss', pos_k=2,
            cls_loss_weight=1.0, x_loss_weight=1.0,
            z_loss_weight=1.0, visibility_loss_weight=1.0,
            aux_loss_weight=0.5, smooth_l1_beta=0.5))


def test_lane_query_decoder_samples_cubic_bezier_at_anchors():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.0)

    predictions = decoder(
        torch.randn(2, 8, 2, 2), torch.nn.Embedding(4, 8))

    assert predictions['curve_x_controls'].shape == (2, 3, 4)
    assert predictions['curve_z_controls'].shape == (2, 3, 4)
    assert predictions['pred_x'].shape == (2, 3, 3)
    assert predictions['pred_z'].shape == (2, 3, 3)
    assert torch.allclose(
        predictions['pred_x'][:, :, 0], predictions['curve_x_controls'][:, :, 0])
    assert torch.allclose(
        predictions['pred_x'][:, :, -1], predictions['curve_x_controls'][:, :, -1])
    assert torch.allclose(
        predictions['pred_z'][:, :, 0], predictions['curve_z_controls'][:, :, 0])
    assert torch.allclose(
        predictions['pred_z'][:, :, -1], predictions['curve_z_controls'][:, :, -1])


def test_decoder_prunes_after_layer_zero_and_keeps_required_training_queries():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=8,
        layer_query_budget=4,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=2, num_heads=2,
        feedforward_channels=16, dropout=0.0)
    required = torch.zeros(1, 8, dtype=torch.bool)
    required[0, 0] = True
    layer_required = torch.zeros(1, 8, dtype=torch.bool)
    layer_required[0, 7] = True

    predictions = decoder(
        torch.randn(1, 8, 2, 2),
        required_query_mask=required,
        required_query_selector=lambda _: layer_required)

    assert predictions['pred_x'].shape[:2] == (1, 4)
    assert predictions['aux_predictions'][0]['pred_x'].shape[:2] == (1, 8)
    assert predictions['selected_query_indices'].shape == (1, 4)
    assert 0 in predictions['selected_query_indices'][0].tolist()
    assert 7 in predictions['selected_query_indices'][0].tolist()
    assert predictions['layer0_topk_indices'].shape == (1, 4)


def test_lane_head_only_forces_refined_queries_into_next_decoder_layer():
    head = _head(8, 2, 2, 4, [0., 2.], 4)
    head.loss_module = LaneLoss()
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.ones(1, 1, 2)
    gt_category = torch.zeros(1, 1, dtype=torch.long)
    anchor_prior = dict(anchor_points=torch.zeros(1, 4, 2, 3))
    refined_required = torch.tensor([[True, False, False, False]])

    with (
        patch.object(head, '_anchor_prior', return_value=anchor_prior),
        patch.object(head, 'forward_bev', return_value=torch.zeros(1, 8, 2, 2)),
        patch.object(head, '_predict', return_value={}) as predict_spy,
        patch.object(head.loss_module, 'forward', return_value={}),
        patch.object(
            head.loss_module, 'required_query_mask',
            return_value=torch.tensor([[False, True, False, False]]),
        ) as initial_required_spy,
        patch.object(
            head.loss_module, 'required_prediction_mask',
            return_value=refined_required,
        ) as refined_required_spy,
        patch.object(head.anchor_prior, 'loss', return_value={}),
        patch.object(head.proposal_expander, 'loss', return_value={}),
    ):
        head.loss(
            None, None, None,
            gt_lanes_3d=gt_lanes,
            gt_lanes_visibility=gt_visibility,
            gt_lanes_category=gt_category)
        predict_kwargs = predict_spy.call_args.kwargs
        refined_selector = predict_kwargs['required_query_selector']
        selected_required = refined_selector(dict(pred_x=torch.empty(0)))

    initial_required_spy.assert_not_called()
    refined_required_spy.assert_called_once()
    assert 'required_query_mask' not in predict_kwargs
    assert selected_required is refined_required


def _selection_predictions(scores, x, visibility):
    score_tensor = torch.tensor([scores])
    x_tensor = torch.tensor([x], dtype=torch.float32)
    visibility_tensor = torch.tensor(
        [visibility], dtype=torch.float32)
    return dict(
        pred_cls=torch.logit(score_tensor),
        pred_x=x_tensor,
        pred_z=torch.zeros_like(x_tensor),
        pred_vis=torch.logit(
            visibility_tensor.clamp(1e-4, 1 - 1e-4)))


def test_lane_head_skips_tube_nms_when_tube_thresholds_are_omitted():
    anchors = [0., 2., 4., 6., 8.]
    head = _head(8, 2, 2, 2, anchors, 4)
    head.score_threshold = .5
    head.curve_nms_cfg = dict(
        visibility_threshold=.5,
        tube_radius=.5,
        z_weight=1.,
        epsilon=1e-6,
        mean_distance_threshold=.35)
    predictions = _selection_predictions(
        scores=[.9, .8],
        x=[[0.] * 5, [0., 0., 0., 0., 2.]],
        visibility=[[1.] * 5] * 2)
    predictions['pred_category'] = torch.zeros(1, 2, 2)

    with (
        patch.object(head, '_anchor_prior', return_value=None),
        patch.object(head, 'forward_bev', return_value=torch.zeros(1, 8, 2, 2)),
        patch.object(head, '_predict', return_value=predictions),
    ):
        output = head.predict(None, None, None)

    assert output['valid_queries'].tolist() == [[True, True]]


def test_decoder_soft_reranking_deprioritizes_a_contained_lower_score_query():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        layer_query_budget=2,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0],
        num_layers=2, num_heads=2, feedforward_channels=16,
        dropout=0.0)
    predictions = _selection_predictions(
        scores=[.9, .85, .8],
        x=[[0.] * 4, [0.] * 4, [3.] * 4],
        visibility=[
            [1.] * 4,
            [0., 1., 1., 0.],
            [1.] * 4,
        ])

    selected, predicted, diagnostics = decoder._select_layer_queries(
        predictions, required_query_mask=None)

    assert predicted.tolist() == [[0, 2]]
    assert selected.tolist() == [[0, 2]]
    assert diagnostics['diag_layer0_soft_rerank_change_ratio'] == .5
    assert diagnostics['diag_layer0_raw_duplicate_pairs'] == 1
    assert diagnostics['diag_layer0_topk_duplicate_pairs'] == 0
    assert diagnostics['layer0_raw_topk_indices'].tolist() == [[0, 1]]


def test_decoder_soft_reranking_keeps_a_better_short_lane_over_overextension():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        layer_query_budget=2,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0],
        num_layers=2, num_heads=2, feedforward_channels=16,
        dropout=0.0)
    predictions = _selection_predictions(
        scores=[.85, .9, .8],
        x=[[0.] * 4, [0.] * 4, [3.] * 4],
        visibility=[
            [1.] * 4,
            [0., 1., 1., 0.],
            [1.] * 4,
        ])

    _, predicted, _ = decoder._select_layer_queries(
        predictions, required_query_mask=None)

    assert predicted.tolist() == [[1, 2]]


def test_decoder_soft_reranking_preserves_lanes_that_diverge():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        layer_query_budget=2,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0],
        num_layers=2, num_heads=2, feedforward_channels=16,
        dropout=0.0)
    predictions = _selection_predictions(
        scores=[.9, .85, .8],
        x=[
            [0., 0., 0., 0.],
            [0., 0., 2., 2.],
            [3., 3., 3., 3.],
        ],
        visibility=[[1.] * 4] * 3)

    _, predicted, _ = decoder._select_layer_queries(
        predictions, required_query_mask=None)

    assert predicted.tolist() == [[0, 1]]


def test_decoder_required_queries_do_not_change_unforced_soft_ranking():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        layer_query_budget=2,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0],
        num_layers=2, num_heads=2, feedforward_channels=16,
        dropout=0.0)
    predictions = _selection_predictions(
        scores=[.9, .8, .1],
        x=[[0.] * 4, [3.] * 4, [6.] * 4],
        visibility=[[1.] * 4] * 3)
    required = torch.tensor([[False, False, True]])

    selected, predicted, _ = decoder._select_layer_queries(
        predictions, required_query_mask=required)

    assert predicted.tolist() == [[0, 1]]
    assert set(selected[0].tolist()) == {0, 2}


def test_decoder_rejects_more_required_queries_than_the_layer_budget():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        layer_query_budget=2,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0],
        num_layers=2, num_heads=2, feedforward_channels=16,
        dropout=0.0)
    predictions = _selection_predictions(
        scores=[.9, .8, .7],
        x=[[0.] * 4, [3.] * 4, [6.] * 4],
        visibility=[[1.] * 4] * 3)

    with pytest.raises(ValueError, match='required queries'):
        decoder._select_layer_queries(
            predictions,
            required_query_mask=torch.ones(1, 3, dtype=torch.bool))


def test_lane_query_decoder_refines_lidar_bezier_controls():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=2,
        anchor_y_steps=[0., 2., 4.], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.)
    x_controls = torch.tensor([[[-2., -2., -1.5, -1.],
                                [2., 2., 2.5, 3.]]])
    z_controls = torch.tensor([[[0., .1, .2, .3],
                                [0., .1, .2, .3]]])

    predictions = decoder(
        torch.randn(1, 8, 2, 2), torch.nn.Embedding(4, 8),
        query_features=torch.randn(1, 2, 8),
        initial_reference_controls=(x_controls, z_controls))

    assert torch.allclose(predictions['curve_x_controls'], x_controls, atol=1e-5)
    assert torch.allclose(predictions['curve_z_controls'], z_controls, atol=1e-5)


def test_lane_query_decoder_preserves_out_of_range_q0_without_shape_clamping():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-2., 0., -2., 2., 4., 2.], num_query=1,
        anchor_y_steps=[0., 2., 4.], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.)
    x_controls = torch.tensor([[[1.5, 2.5, 3.5, 4.5]]])
    z_controls = torch.tensor([[[0., .5, 1., 2.5]]])

    predictions = decoder(
        torch.randn(1, 8, 2, 2),
        initial_reference_controls=(x_controls, z_controls))

    assert torch.allclose(predictions['curve_x_controls'], x_controls)
    assert torch.allclose(predictions['curve_z_controls'], z_controls)


def test_bev_encoder_keeps_lidar_bev_without_constructing_image_bev():
    head = _head(16, 4, 3, 3, [0., 2., 4.], 8)
    image_features = [torch.randn(2, 1, 16, 8, 8)]
    lidar_features = [torch.randn(2, 8, 2, 2)]
    metas = _metas(2, image_size=8)

    image_bev = head.forward_image_bev(image_features, metas)
    lidar_bev = head.bev_encoder.forward_lidar_bev(lidar_features)
    fused_call = head.forward_bev(image_features, lidar_features, metas)

    assert image_bev is None
    assert lidar_bev.shape == (2, 16, 4, 3)
    assert torch.allclose(fused_call, lidar_bev)


def test_lane_query_decoder_exposes_each_intermediate_layer_for_auxiliary_loss():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=3, num_heads=2,
        feedforward_channels=16, dropout=0.0)

    predictions = decoder(
        torch.randn(2, 8, 2, 2), torch.nn.Embedding(4, 8))

    assert len(predictions['aux_predictions']) == 2
    assert all(
        auxiliary['pred_x'].shape == (2, 3, 3)
        for auxiliary in predictions['aux_predictions'])
    assert predictions['pred_x'].shape == (2, 3, 3)
    assert torch.allclose(
        predictions['curve_x_controls'],
        predictions['aux_predictions'][0]['curve_x_controls'], atol=1e-5)
    assert torch.allclose(
        predictions['curve_z_controls'],
        predictions['aux_predictions'][0]['curve_z_controls'], atol=1e-5)


def test_lane_query_decoder_composes_nonzero_residual_with_previous_controls():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=2, num_heads=2,
        feedforward_channels=16, dropout=0.0)
    residual = 0.4
    with torch.no_grad():
        decoder.x_refine_heads[0].bias.fill_(residual)

    predictions = decoder(
        torch.randn(2, 8, 2, 2), torch.nn.Embedding(4, 8))
    previous = predictions['aux_predictions'][0]['curve_x_controls']
    expected = previous + torch.tanh(torch.tensor(residual)) * 5.0

    assert torch.allclose(predictions['curve_x_controls'], expected, atol=1e-5)
    assert not torch.allclose(predictions['curve_x_controls'], previous)


def test_lane_query_decoder_predicts_an_ordered_visibility_interval():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.0,
        visibility_start_prior=1 / 3, visibility_end_prior=2 / 3)

    predictions = decoder(
        torch.randn(2, 8, 2, 2), torch.nn.Embedding(4, 8))

    assert predictions['pred_vis_start'].shape == (2, 3)
    assert predictions['pred_vis_end'].shape == (2, 3)
    assert predictions['pred_vis_interval'].shape == (2, 3, 4)
    assert torch.all(predictions['pred_vis_start'] <= predictions['pred_vis_end'])
    assert torch.allclose(
        predictions['pred_vis_start'], torch.ones(2, 3), atol=1e-5)
    assert torch.allclose(
        predictions['pred_vis_end'], torch.full((2, 3), 2.0), atol=1e-5)
    assert torch.all(predictions['pred_vis_interval'] >= 0)
    assert torch.all(predictions['pred_vis_interval'] <= 1)


def test_lane_query_decoder_refines_visibility_bounds_between_layers():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0, 6.0], num_layers=2, num_heads=2,
        feedforward_channels=16, dropout=0.0,
        visibility_start_prior=1 / 3, visibility_end_prior=2 / 3)
    residual = torch.tensor([0.4, -0.3])
    with torch.no_grad():
        decoder.vis_bounds_refine_heads[0].bias.copy_(residual)

    predictions = decoder(
        torch.randn(2, 8, 2, 2), torch.nn.Embedding(4, 8))
    previous = predictions['aux_predictions'][0]
    previous_start = previous['pred_vis_start']
    previous_end = previous['pred_vis_end']
    count = decoder.num_anchors
    normalized_start = ((previous_start + 0.5) / count).clamp(1e-4, 1 - 1e-4)
    normalized_span = (
        (previous_end - previous_start)
        / (count - 0.5 - previous_start)).clamp(1e-4, 1 - 1e-4)
    expected_start = (
        torch.sigmoid(torch.logit(normalized_start) + residual[0]) * count - 0.5)
    expected_end = expected_start + torch.sigmoid(
        torch.logit(normalized_span) + residual[1]) * (
            count - 0.5 - expected_start)

    assert torch.allclose(predictions['pred_vis_start'], expected_start, atol=1e-5)
    assert torch.allclose(predictions['pred_vis_end'], expected_end, atol=1e-5)
    assert torch.allclose(
        predictions['pred_vis'],
        torch.logit(predictions['pred_vis_interval'].clamp(1e-4, 1 - 1e-4)))


def test_two_layer_decoder_resamples_refined_curve_for_second_layer():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=2, num_heads=2,
        feedforward_channels=16, dropout=0.0)
    sampled = []

    def sample_image_features(points, query_features):
        assert query_features.shape == (2, 3, 8)
        sampled.append(points.clone())
        return points.new_zeros(points.size(0), points.size(1), 8)

    bev = torch.randn(2, 8, 2, 2)
    bev_pos = torch.nn.Embedding(4, 8)
    initial_x = torch.tensor([[[-2., -2., -1., 0.]]]).expand(2, 3, -1)
    initial_z = torch.zeros_like(initial_x)
    predictions = decoder(
        bev, bev_pos, initial_reference_controls=(initial_x, initial_z),
        query_feature_sampler=sample_image_features)
    first_layer = predictions['aux_predictions'][0]

    assert len(sampled) == 2
    assert torch.allclose(sampled[0][..., 0], torch.einsum(
        'ac,bqc->bqa', decoder.bezier_basis, initial_x))
    assert torch.allclose(sampled[1][..., 0], first_layer['pred_x'])
    assert torch.allclose(sampled[1][..., 2], first_layer['pred_z'])
    assert first_layer['pred_logits'].shape == predictions['pred_logits'].shape


def test_decoder_cross_attention_keeps_each_query_evidence_separate():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=2,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.0)
    layer = decoder.layers[0]
    with torch.no_grad():
        layer.self_attn.out_proj.weight.zero_()
        layer.self_attn.out_proj.bias.zero_()
        layer.linear2.weight.zero_()
        layer.linear2.bias.zero_()
    controls = torch.zeros(1, 2, 4)

    def sample(value):
        def sampler(points, query_features):
            assert query_features.shape == (1, 2, 8)
            tokens = points.new_zeros(1, 2, 1, 8)
            tokens[:, 0] = value
            return dict(
                tokens=tokens,
                key_padding_mask=torch.zeros(
                    1, 2, 1, dtype=torch.bool, device=points.device))
        return sampler

    baseline = decoder(
        torch.randn(1, 8, 2, 2),
        initial_reference_controls=(controls, controls),
        query_feature_sampler=sample(0.0))
    changed = decoder(
        torch.randn(1, 8, 2, 2),
        initial_reference_controls=(controls, controls),
        query_feature_sampler=sample(5.0))

    assert not torch.allclose(
        baseline['pred_logits'][:, 0], changed['pred_logits'][:, 0])
    assert torch.allclose(
        baseline['pred_logits'][:, 1], changed['pred_logits'][:, 1])


def test_final_layer_loss_reaches_layer_zero_geometry_through_q1():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=2,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=2, num_heads=2,
        feedforward_channels=16, dropout=0.0)
    controls = torch.zeros(1, 2, 4)

    predictions = decoder(
        torch.randn(1, 8, 2, 2),
        initial_reference_controls=(controls, controls))
    predictions['pred_x'].sum().backward()

    assert decoder.x_reference_head.weight.grad is not None
    assert decoder.x_reference_head.weight.grad.abs().sum() > 0


def test_lane_head_samples_initial_and_layer_zero_refined_anchors():
    head = _head(16, 4, 3, 3, [0., 2., 4.], 8)
    original = head.query_sampler.forward
    head.query_sampler.forward = Mock(wraps=original)

    head.predict(
        [torch.randn(1, 1, 16, 8, 8)],
        [torch.randn(1, 8, 2, 2)], _metas(1, image_size=8))

    assert head.query_sampler.forward.call_count == 2


def test_lane_query_decoder_initializes_objectness_with_a_low_prior():
    decoder = LaneQueryDecoder(
        embed_dims=8,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.0,
        objectness_prior_probability=0.01)

    assert torch.allclose(
        decoder.cls_head.bias.sigmoid(), torch.tensor([0.01]), atol=1e-6)


def test_lane_query_decoder_rejects_disabled_denoising_branch():
    with pytest.raises(ValueError, match='denoising queries are not supported'):
        LaneQueryDecoder(
            embed_dims=8,
            pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
            anchor_y_steps=[0.0, 2.0, 4.0], num_layers=1, num_heads=2,
            feedforward_channels=16, dropout=0.0, dn_num_groups=1)


def test_lane_head_degrades_gracefully_when_lidar_features_are_empty():
    head = _head(
        16, 4, 3, 6, [0., 2., 4.], 8, max_templates=3)

    output = head.predict(
        [torch.randn(1, 3, 1, 16, 8, 8)], None,
        _metas(1, image_size=8))

    assert output['x'].shape == (1, 6, 3)
    assert output['shape_templates'].shape == (1, 3, 4, 3)
    assert output['shape_lateral_slopes'].shape == (1, 3, 3)
    assert output['shape_vertical_slopes'].shape == (1, 3, 3)
    assert output['shape_control_yaw_degrees'].shape == (1, 3, 3)
    assert output['shape_control_pitch_degrees'].shape == (1, 3, 3)
    assert output['unique_template_counts'].tolist() == [1]
    assert 'road_curvature' not in output
    assert 'road_grade' not in output


def test_lane_head_uses_51_anchors_and_backpropagates():
    torch.manual_seed(0)
    anchors = [2.0 * index for index in range(51)]
    head = _head(
        32, 8, 5, 50, anchors, 16, max_templates=5)
    batch_size = 2
    image_features = [torch.randn(batch_size, 3, 1, 32, 16, 16)]
    lidar_features = [torch.randn(batch_size, 16, 4, 4)]
    metas = _metas(batch_size)

    gt_lanes = torch.zeros(batch_size, 20, 51, 3)
    gt_lanes[:, :, :, 1] = torch.tensor(anchors)
    gt_visibility = torch.zeros(batch_size, 20, 51)
    anchor_tensor = torch.tensor(anchors)
    gt_lanes[:, 0, :, 0] = -2.0 + 0.5 * 0.002 * anchor_tensor.square()
    gt_lanes[:, 0, :, 2] = 0.02 * anchor_tensor
    gt_visibility[:, 0, :30] = 1.0
    gt_category = torch.zeros(batch_size, 20, dtype=torch.long)
    gt_category[:, 0] = 2
    enhanced_lanes = gt_lanes.clone()
    enhanced_visibility = gt_visibility.clone()
    enhanced_lanes[:, 0, 30:35, 0] = -1.5
    enhanced_visibility[:, 0, 30:35] = 1.0
    prior_loss = head.anchor_prior.loss
    expander_loss = head.proposal_expander.loss
    with (
        patch.object(head.anchor_prior, 'loss', wraps=prior_loss) as prior_spy,
        patch.object(
            head.proposal_expander, 'loss',
            wraps=expander_loss) as expander_spy,
    ):
        losses = head.loss(
            image_features, lidar_features, metas,
            gt_lanes_3d=gt_lanes, gt_lanes_visibility=gt_visibility,
            gt_lanes_category=gt_category,
            gt_lanes_3d_enhanced=enhanced_lanes,
            gt_lanes_visibility_enhanced=enhanced_visibility)
    assert prior_spy.call_args.args[1] is enhanced_lanes
    assert prior_spy.call_args.args[2] is enhanced_visibility
    assert expander_spy.call_args.args[1] is enhanced_lanes
    assert expander_spy.call_args.args[2] is enhanced_visibility
    total_loss = sum(losses.values())
    total_loss.backward()

    assert head.anchor_y_steps.numel() == 51
    final_lane_loss_names = {
        'loss_cls', 'loss_x', 'loss_z', 'loss_visibility'}
    auxiliary_loss_names = {
        f'loss_aux_0_{name.removeprefix("loss_")}'
        for name in final_lane_loss_names}
    lidar_loss_names = {
        'loss_lidar_shape_cover', 'loss_lidar_shape_mode',
        'loss_lidar_shape_diversity',
        'loss_q0_geometry', 'diag_q0_gt_recall',
        'diag_q0_best_distance',
        'diag_predicted_shape_metric_rank', 'diag_gt_shape_metric_rank',
        'diag_shape_mode_cover_recall'}
    assert set(losses) == (
        final_lane_loss_names | auxiliary_loss_names | lidar_loss_names)
    assert torch.isfinite(total_loss)
    assert head.lane_decoder.query.weight.grad is not None
    assert torch.isfinite(head.lane_decoder.query.weight.grad).all()
    assert head.anchor_prior.template_head[-1].weight.grad is not None
    assert torch.isfinite(
        head.anchor_prior.template_head[-1].weight.grad).all()
    assert head.proposal_expander.template_adapter.weight.grad is not None
    assert torch.isfinite(
        head.proposal_expander.template_adapter.weight.grad).all()
    assert head.query_sampler.image_adapter.weight.grad is not None
    assert torch.isfinite(head.query_sampler.image_adapter.weight.grad).all()


def test_topk_assigner_selects_two_exclusive_anchors_per_gt():
    assigner = TopkAssigner(pos_k=2)
    anchor_x = torch.tensor([
        [-2.2, -2.2], [-1.8, -1.8], [-1.4, -1.4],
        [1.4, 1.4], [1.8, 1.8], [2.2, 2.2]])
    anchor_z = torch.zeros_like(anchor_x)
    gt_x = torch.tensor([[-2., -2.], [2., 2.]])
    gt_z = torch.zeros_like(gt_x)
    gt_visibility = torch.ones(2, 2)

    assigned = assigner.assign(
        anchor_x, anchor_z, gt_x, gt_z, gt_visibility)

    assert (assigned == 0).sum() == 2
    assert (assigned == 1).sum() == 2
    assert (assigned == -1).sum() == 2


def test_hungarian_assigner_gives_every_gt_one_exclusive_query():
    assigner = HungarianAssigner()
    anchor_x = torch.tensor([[0., 0.], [.1, .1], [20., 20.]])
    anchor_z = torch.zeros_like(anchor_x)
    gt_x = torch.tensor([[0., 0.], [10., 10.]])
    gt_z = torch.zeros_like(gt_x)
    gt_visibility = torch.ones(2, 2)

    assigned = assigner.assign(
        anchor_x, anchor_z, gt_x, gt_z, gt_visibility)

    assert (assigned == 0).sum() == 1
    assert (assigned == 1).sum() == 1
    assert (assigned == -1).sum() == 1


def test_one_to_one_lane_loss_uses_hungarian_assignment():
    loss_module = LaneLoss()

    assert isinstance(loss_module.assigner, HungarianAssigner)
    assert loss_module.emit_diagnostics is True


def test_one_to_one_loss_separates_lane_score_from_positive_only_category():
    loss_module = LaneLoss(
        negative_objectness_weight=.1)
    predictions = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.]]]),
        pred_z=torch.zeros(1, 2, 2),
        pred_vis=torch.zeros(1, 2, 2, requires_grad=True),
        pred_cls=torch.zeros(1, 2, requires_grad=True),
        pred_category=torch.zeros(1, 2, 21, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.ones(1, 1, 2)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=torch.tensor([[2]]))
    (losses['loss_score'] + losses['loss_category']).backward()

    assert 'loss_cls' not in losses
    assert losses['loss_score'] > 0
    assert losses['loss_category'] > 0
    assert predictions['pred_cls'].grad.abs().sum() > 0
    assert predictions['pred_category'].grad[0, 0].abs().sum() > 0
    assert torch.all(predictions['pred_category'].grad[0, 1] == 0)


def test_lane_score_uses_detached_soft_tube_iou_for_matched_queries():
    loss_module = LaneLoss(
        negative_objectness_weight=1.0,
        ambiguous_negative_weight=1.0,
        tube_loss_weight=0.5)
    score_logits = torch.zeros(1, 2, requires_grad=True)
    predicted_x = torch.tensor(
        [[[0.0, 0.0], [10.0, 10.0]]], requires_grad=True)
    predicted_visibility = torch.full(
        (1, 2, 2), torch.logit(torch.tensor(0.25)),
        requires_grad=True)
    predictions = dict(
        pred_x=predicted_x,
        pred_z=torch.zeros(1, 2, 2, requires_grad=True),
        pred_vis=predicted_visibility,
        pred_cls=score_logits,
        pred_category=torch.zeros(1, 2, 21, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 2, 3)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=torch.ones(1, 1, 2),
        gt_lanes_category=torch.zeros(1, 1, dtype=torch.long))
    losses['loss_score'].backward()

    assert torch.allclose(
        score_logits.grad, torch.tensor([[0.25, 0.5]]), atol=1e-5)
    assert predicted_x.grad is None
    assert predicted_visibility.grad is None


def test_tube_loss_updates_positive_geometry_and_visibility():
    loss_module = LaneLoss(tube_loss_weight=0.5)
    predicted_x = torch.full(
        (1, 1, 3), 0.2, requires_grad=True)
    predicted_visibility = torch.zeros(
        1, 1, 3, requires_grad=True)
    predictions = dict(
        pred_x=predicted_x,
        pred_z=torch.zeros(1, 1, 3, requires_grad=True),
        pred_vis=predicted_visibility,
        pred_cls=torch.zeros(1, 1, requires_grad=True),
        pred_category=torch.zeros(1, 1, 21, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 3, 3)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=torch.ones(1, 1, 3),
        gt_lanes_category=torch.zeros(1, 1, dtype=torch.long))
    losses['loss_tube'].backward()

    assert losses['loss_tube'] > 0
    assert predicted_x.grad is not None
    assert predicted_x.grad.abs().sum() > 0
    assert predicted_visibility.grad is not None
    assert predicted_visibility.grad.abs().sum() > 0


def test_balanced_lane_score_loss_is_invariant_to_easy_negative_count():
    loss_module = LaneLoss(
        negative_objectness_weight=.1)

    def score_loss(num_queries):
        predictions = dict(
            pred_x=torch.cat((
                torch.zeros(1, 1, 2),
                torch.full((1, num_queries - 1, 2), 10.)), dim=1),
            pred_z=torch.zeros(1, num_queries, 2),
            pred_vis=torch.zeros(1, num_queries, 2),
            pred_cls=torch.zeros(1, num_queries),
            pred_category=torch.zeros(1, num_queries, 21))
        gt_lanes = torch.zeros(1, 1, 2, 3)
        return loss_module(
            predictions, gt_lanes_3d=gt_lanes,
            gt_lanes_visibility=torch.ones(1, 1, 2),
            gt_lanes_category=torch.zeros(1, 1, dtype=torch.long),
        )['loss_score']

    assert torch.allclose(score_loss(4), score_loss(40))


def test_geometrically_ambiguous_unmatched_query_is_downweighted():
    loss_module = LaneLoss(
        negative_objectness_weight=1.0,
        ambiguous_negative_weight=.1,
        ambiguous_distance_threshold=.5,
        ambiguous_match_ratio=.75)
    pred_cls = torch.zeros(1, 3, requires_grad=True)
    predictions = dict(
        pred_x=torch.tensor([[[0., 0.], [.1, .1], [10., 10.]]]),
        pred_z=torch.zeros(1, 3, 2),
        pred_vis=torch.zeros(1, 3, 2),
        pred_cls=pred_cls,
        pred_category=torch.zeros(1, 3, 21))
    gt_lanes = torch.zeros(1, 1, 2, 3)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=torch.ones(1, 1, 2),
        gt_lanes_category=torch.zeros(1, 1, dtype=torch.long))
    losses['loss_score'].backward()

    assert pred_cls.grad[0, 1].abs() < pred_cls.grad[0, 2].abs() * .2


def test_auxiliary_ambiguous_weight_defaults_to_final_weight():
    loss_module = LaneLoss(
        negative_objectness_weight=1.0,
        ambiguous_negative_weight=.4,
        aux_loss_weight=1.0)

    assert loss_module.auxiliary_ambiguous_negative_weight == .4


def test_auxiliary_ambiguous_weight_can_be_configured_independently():
    loss_module = LaneLoss(
        negative_objectness_weight=1.0,
        ambiguous_negative_weight=1.0,
        auxiliary_ambiguous_negative_weight=.1,
        aux_loss_weight=1.0)
    auxiliary = dict(
        pred_x=torch.tensor([[[0., 0.], [.1, .1], [10., 10.]]]),
        pred_z=torch.zeros(1, 3, 2),
        pred_vis=torch.zeros(1, 3, 2),
        pred_cls=torch.zeros(1, 3),
        pred_category=torch.zeros(1, 3, 21))
    predictions = dict(auxiliary)
    predictions['aux_predictions'] = [auxiliary]
    gt_lanes = torch.zeros(1, 1, 2, 3)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=torch.ones(1, 1, 2),
        gt_lanes_category=torch.zeros(1, 1, dtype=torch.long))

    assert loss_module.auxiliary_ambiguous_negative_weight == .1
    assert losses['loss_aux_0_score'] < losses['loss_score']


def test_layer_zero_topk_recall_uses_unforced_ranking():
    loss_module = LaneLoss()
    auxiliary = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.], [20., 20.]]]),
        pred_z=torch.zeros(1, 3, 2),
        pred_vis=torch.zeros(1, 3, 2),
        pred_cls=torch.zeros(1, 3),
        pred_category=torch.zeros(1, 3, 21))
    predictions = dict(
        pred_x=auxiliary['pred_x'][:, :2],
        pred_z=auxiliary['pred_z'][:, :2],
        pred_vis=auxiliary['pred_vis'][:, :2],
        pred_cls=auxiliary['pred_cls'][:, :2],
        pred_category=auxiliary['pred_category'][:, :2],
        aux_predictions=[auxiliary],
        layer0_topk_indices=torch.tensor([[0, 2]]),
        selected_query_indices=torch.tensor([[0, 1]]),
        diag_layer0_soft_rerank_change_ratio=torch.tensor(.5),
        diag_layer0_raw_duplicate_pairs=torch.tensor(1.))
    gt_lanes = torch.zeros(1, 2, 2, 3)
    gt_lanes[0, 1, :, 0] = 10.

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=torch.ones(1, 2, 2),
        gt_lanes_category=torch.zeros(1, 2, dtype=torch.long))

    assert losses['diag_layer0_topk_gt_recall'] == .5
    assert losses['diag_layer0_training_gt_recall'] == 1
    assert losses['diag_layer0_topk_queries_per_gt'] == .5
    assert losses['diag_layer0_topk_queries_per_gt_max'] == 1
    assert losses['diag_layer0_topk_uncovered_gt_ratio'] == .5
    assert losses['diag_layer0_soft_rerank_change_ratio'] == .5
    assert losses['diag_layer0_raw_duplicate_pairs'] == 1


def test_one_to_one_lane_loss_reports_background_collapse_diagnostics():
    loss_module = LaneLoss()
    predictions = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.]]]),
        pred_z=torch.zeros(1, 2, 2),
        pred_vis=torch.full((1, 2, 2), 10.),
        pred_cls=torch.full((1, 2), -10.),
        pred_category=torch.zeros(1, 2, 21),
        pred_logits=torch.cat((
            torch.zeros(1, 2, 1),
            torch.full((1, 2, 21), -10.)), dim=-1))
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.ones(1, 1, 2)
    gt_category = torch.zeros(1, 1, dtype=torch.long)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category)

    assert losses['diag_gt_match_recall'] == 1
    assert losses['diag_unmatched_gt'] == 0
    assert losses['diag_duplicate_gt_assignments'] == 0
    assert losses['diag_positive_queries'] == 1
    assert losses['diag_lane_score_queries'] == 0
    assert losses['diag_prediction_queries'] == 0
    assert losses['diag_pred_gt_ratio'] == 0
    assert losses['diag_positive_lane_score'] < 1e-4
    assert losses['diag_negative_lane_score'] < 1e-4
    assert losses['diag_lane_score_margin'] == 0
    assert losses['diag_positive_tube_iou'] > .99


def test_one_to_one_lane_loss_prediction_diagnostics_match_inference_filter():
    loss_module = LaneLoss()
    predictions = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.], [20., 20.]]]),
        pred_z=torch.zeros(1, 3, 2),
        pred_vis=torch.tensor([[
            [10., 10.], [10., 10.], [-10., -10.]]]),
        pred_cls=torch.tensor([[10., -10., 10.]]),
        pred_category=torch.zeros(1, 3, 21),
        pred_logits=torch.tensor([[
            [0., 10.] + [-10.] * 20,
            [0.] + [-10.] * 21,
            [0., 10.] + [-10.] * 20,
        ]]))
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.ones(1, 1, 2)
    gt_category = torch.zeros(1, 1, dtype=torch.long)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category)

    # Query 0 passes objectness and visibility even while category is
    # uncertain. Query 2 has the same objectness but no visible point.
    assert losses['diag_lane_score_queries'] == 2
    assert losses['diag_prediction_queries'] == 1
    assert losses['diag_pred_gt_ratio'] == 1


def test_lane_loss_reports_tube_quality_and_lane_score_margin():
    loss_module = LaneLoss()
    predictions = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.]]]),
        pred_z=torch.zeros(1, 2, 2),
        pred_vis=torch.zeros(1, 2, 2),
        pred_cls=torch.logit(torch.tensor([[.8, .2]])),
        pred_category=torch.zeros(1, 2, 21))
    gt_lanes = torch.zeros(1, 1, 2, 3)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=torch.ones(1, 1, 2),
        gt_lanes_category=torch.zeros(1, 1, dtype=torch.long))

    assert torch.allclose(
        losses['diag_positive_lane_score'], torch.tensor(.8))
    assert torch.allclose(
        losses['diag_negative_lane_score'], torch.tensor(.2))
    assert torch.allclose(
        losses['diag_lane_score_margin'], torch.tensor(.6))
    assert torch.allclose(
        losses['diag_positive_tube_iou'], torch.tensor(.5), atol=1e-5)


def test_one_to_one_lane_loss_matches_final_predictions_not_q0():
    loss_module = LaneLoss()
    predictions = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.]]]),
        pred_z=torch.zeros(1, 2, 2),
        anchor_points=torch.tensor([[
            [[0., 0., 0.], [0., 1., 0.]],
            [[.1, 0., 0.], [.1, 1., 0.]],
        ]]))
    gt_lanes = torch.zeros(1, 2, 2, 3)
    gt_lanes[0, 1, :, 0] = 10.
    gt_visibility = torch.ones(1, 2, 2)

    assignment = loss_module._assign(
        predictions, gt_lanes, gt_visibility)[0]

    assert assignment.tolist() == [0, 1]


def test_one_to_one_lane_loss_matches_each_decoder_layer_independently():
    loss_module = LaneLoss()
    gt_lanes = torch.zeros(1, 2, 2, 3)
    gt_lanes[0, 1, :, 0] = 10.
    gt_visibility = torch.ones(1, 2, 2)
    final = dict(
        pred_x=torch.tensor([[[0., 0.], [10., 10.]]]),
        pred_z=torch.zeros(1, 2, 2))
    auxiliary = dict(
        pred_x=torch.tensor([[[10., 10.], [0., 0.]]]),
        pred_z=torch.zeros(1, 2, 2))
    final_assignment = loss_module._assign(
        final, gt_lanes, gt_visibility)

    auxiliary_assignment = loss_module._auxiliary_assignments(
        auxiliary, final_assignment, gt_lanes, gt_visibility)

    assert final_assignment[0].tolist() == [0, 1]
    assert auxiliary_assignment[0].tolist() == [1, 0]


def test_one_to_one_lane_loss_reports_auxiliary_matching_health():
    loss_module = LaneLoss()

    def prediction(x):
        return dict(
            pred_x=x,
            pred_z=torch.zeros_like(x),
            pred_vis=torch.full_like(x, 10.),
            pred_cls=torch.full(x.shape[:2], -10.),
            pred_category=torch.zeros(*x.shape[:2], 21),
            pred_logits=torch.cat((
                torch.zeros(*x.shape[:2], 1),
                torch.full((*x.shape[:2], 21), -10.)), dim=-1))

    predictions = prediction(torch.tensor([[[0., 0.], [10., 10.]]]))
    predictions['aux_predictions'] = [
        prediction(torch.tensor([[[10., 10.], [0., 0.]]]))]
    gt_lanes = torch.zeros(1, 2, 2, 3)
    gt_lanes[0, 1, :, 0] = 10.
    gt_visibility = torch.ones(1, 2, 2)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=torch.zeros(1, 2, dtype=torch.long))

    assert losses['diag_aux_0_gt_match_recall'] == 1
    assert losses['diag_aux_0_unmatched_gt'] == 0
    assert losses['diag_aux_0_duplicate_gt_assignments'] == 0


def test_topk_lane_loss_has_one_visibility_objective():
    loss_module = TopkLaneLoss(pos_k=2)
    anchor_x = torch.tensor([[-1.5, -1.5], [-.5, -.5],
                             [.5, .5], [1.5, 1.5]])
    anchor_points = torch.stack((
        anchor_x, torch.tensor([[0., 2.]]).expand_as(anchor_x),
        torch.zeros_like(anchor_x)), dim=-1).unsqueeze(0)
    predictions = dict(
        pred_x=anchor_x.unsqueeze(0).requires_grad_(),
        pred_z=torch.zeros(1, 4, 2, requires_grad=True),
        pred_vis=torch.zeros(1, 4, 2, requires_grad=True),
        pred_logits=torch.zeros(1, 4, 22, requires_grad=True),
        anchor_points=anchor_points)
    gt_lanes = torch.zeros(1, 2, 2, 3)
    gt_lanes[0, :, :, 1] = torch.tensor([0., 2.])
    gt_lanes[0, 0, :, 0] = -1.
    gt_lanes[0, 1, :, 0] = 1.
    gt_visibility = torch.tensor([[[1., 1.], [1., 0.]]])
    gt_category = torch.tensor([[2, 7]])

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category)
    sum(losses.values()).backward()

    assert set(losses) == {'loss_cls', 'loss_x', 'loss_z', 'loss_visibility'}
    assert predictions['pred_vis'].grad is not None
    assert torch.isfinite(sum(losses.values()))



def test_empty_gt_keeps_z_prediction_branch_in_loss_graph():
    head = _head(16, 2, 2, 2, [0.0, 2.0], 16)
    predictions = dict(
        pred_x=torch.randn(1, 2, 2, requires_grad=True),
        pred_z=torch.randn(1, 2, 2, requires_grad=True),
        pred_vis=torch.randn(1, 2, 2, requires_grad=True),
        pred_cls=torch.randn(1, 2, requires_grad=True),
        pred_category=torch.randn(1, 2, 21, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.zeros(1, 1, 2)
    gt_category = torch.zeros(1, 1, dtype=torch.long)

    loss = sum(head.loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category).values())
    loss.backward()

    assert predictions['pred_z'].grad is not None
    assert torch.all(predictions['pred_z'].grad == 0)
