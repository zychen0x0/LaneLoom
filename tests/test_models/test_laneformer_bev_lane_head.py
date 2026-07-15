import torch

from LaneFormer.models.head.laneformer_bev_lane_head import LaneFormerBEVLaneHead
from LaneFormer.models.head.laneformer_lane_query_decoder import (
    LaneFormerLaneQueryDecoder,
)
from LaneFormer.models.head.lidar_bezier_anchor_prior import (
    LaneFormerLiDARBezierAnchorPrior,
)
from LaneFormer.models.losses.laneformer_hungarian_loss import (
    LaneFormerHungarianLaneLoss,
)


def _metas(batch_size, image_size=16):
    lidar2img = torch.tensor([
        [1.0, 0.0, 0.0, image_size / 2],
        [0.0, 1.0, 0.0, image_size / 2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return [dict(
        current=dict(lidar2img=[lidar2img], img_shape=(image_size, image_size),
                     lidaradj2lidarcurr=torch.eye(4)))
        for _ in range(batch_size)]


def _head(embed_dims, bev_h, bev_w, num_query, anchors, lidar_channels,
          dn_num_groups=0):
    return LaneFormerBEVLaneHead(
        anchor_prior=dict(
            type='LaneFormerLiDARBezierAnchorPrior',
            lidar_in_channels=lidar_channels,
            image_in_channels=embed_dims,
            embed_dims=embed_dims,
            pc_range=[-25., 0., -5., 25., 100., 5.],
            anchor_y_steps=anchors,
            num_anchors=num_query,
            hidden_channels=8),
        bev_encoder=dict(
            type='LaneFormerBEVEncoder', embed_dims=embed_dims,
            bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.],
            spatial_cross_attn=dict(
                type='LaneFormerSpatialCrossAttention', embed_dims=embed_dims,
                bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.],
                num_points_in_pillar=4, num_levels=1),
            lidar_bev_adapter=dict(
                type='LaneFormerLiDARBEVAdapter', embed_dims=embed_dims,
                lidar_in_channels=lidar_channels),
            lidar_fusion=dict(
                type='LaneFormerImageLiDARFusion', embed_dims=embed_dims,
                num_layers=1, num_heads=4,
                feedforward_channels=embed_dims * 2, dropout=0.0),
            temporal_self_attn=dict(
                type='LaneFormerTemporalSelfAttention', embed_dims=embed_dims,
                bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.])),
        lane_decoder=dict(
            type='LaneFormerLaneQueryDecoder', embed_dims=embed_dims,
            bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.],
            num_query=num_query, anchor_y_steps=anchors, num_layers=2,
            num_heads=4, feedforward_channels=embed_dims * 2, dropout=0.0,
            dn_num_groups=dn_num_groups),
        loss_module=dict(
            type='LaneFormerHungarianLaneLoss', geometry_loss_weight=20.0,
            visibility_loss_weight=1.0, objectness_loss_weight=10.0,
            category_loss_weight=0.25, geometry_huber_beta=0.5))


def test_lane_query_decoder_samples_cubic_bezier_at_anchors():
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
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


def test_lidar_prior_generates_one_metre_bezier_anchors_and_samples_images():
    prior = LaneFormerLiDARBezierAnchorPrior(
        lidar_in_channels=8, image_in_channels=8, embed_dims=8,
        pc_range=[-2., 0., -2., 2., 4., 2.],
        anchor_y_steps=[0., 2., 4.], num_anchors=4,
        hidden_channels=4, max_curvature=.02, max_grade=.2)
    with torch.no_grad():
        prior.geometry_encoder[-1].bias.copy_(torch.atanh(torch.tensor([.5, .25])))
    projection = torch.tensor([
        [1., 0., 0., 4.],
        [0., 1., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
    ])

    output = prior(
        [torch.randn(1, 8, 3, 3)],
        [torch.randn(1, 1, 8, 8, 8)],
        [dict(current=dict(lane2img=[projection], img_shape=(8, 8)))])

    assert torch.allclose(prior.start_x, torch.tensor([-1.5, -.5, .5, 1.5]))
    assert torch.allclose(output['curvature'], torch.tensor([.01]), atol=1e-6)
    assert torch.allclose(output['grade'], torch.tensor([.05]), atol=1e-6)
    assert torch.allclose(
        output['curve_x_controls'][0, :, 2], prior.start_x + .01 * 16 / 6,
        atol=1e-6)
    assert torch.allclose(
        output['curve_x_controls'][0, :, 3], prior.start_x + .01 * 16 / 2,
        atol=1e-6)
    assert torch.allclose(
        output['curve_z_controls'][0, 0],
        torch.tensor([0., .05 * 4 / 3, .05 * 8 / 3, .2]), atol=1e-6)
    assert output['anchor_points'].shape == (1, 4, 3, 3)
    assert output['query_features'].shape == (1, 4, 8)
    assert torch.all(output['projection_valid_ratio'] > 0)


def test_lidar_prior_fits_curvature_and_grade_from_lane_gt():
    prior = LaneFormerLiDARBezierAnchorPrior(
        lidar_in_channels=8, image_in_channels=8, embed_dims=8,
        pc_range=[-2., 0., -2., 2., 6., 2.],
        anchor_y_steps=[0., 2., 4., 6.], num_anchors=4,
        hidden_channels=4, max_curvature=.02, max_grade=.2)
    y = torch.tensor([0., 2., 4., 6.])
    curvature, grade = .006, .04
    lanes = torch.zeros(1, 2, 4, 3)
    lanes[0, :, :, 1] = y
    lanes[0, 0, :, 0] = -1 + .5 * curvature * y.square()
    lanes[0, 1, :, 0] = 1 + .5 * curvature * y.square()
    lanes[0, :, :, 2] = .2 + grade * y
    visibility = torch.ones(1, 2, 4)

    target_curvature, target_grade, valid = prior.fit_targets(
        lanes, visibility)

    assert valid.tolist() == [True]
    assert torch.allclose(target_curvature, torch.tensor([curvature]), atol=1e-5)
    assert torch.allclose(target_grade, torch.tensor([grade]), atol=1e-5)


def test_lane_query_decoder_refines_lidar_bezier_controls():
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=2,
        anchor_y_steps=[0., 2., 4.], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.)
    with torch.no_grad():
        decoder.x_head.weight.zero_()
        decoder.x_head.bias.zero_()
        decoder.z_head.weight.zero_()
        decoder.z_head.bias.zero_()
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


def test_bev_encoder_exposes_independent_image_and_lidar_adapters():
    head = _head(16, 4, 3, 3, [0., 2., 4.], 8)
    image_features = [torch.randn(2, 1, 16, 8, 8)]
    lidar_features = [torch.randn(2, 8, 2, 2)]
    metas = _metas(2, image_size=8)

    image_bev = head.forward_image_bev(image_features, metas)
    lidar_bev = head.bev_encoder.forward_lidar_bev(lidar_features)
    image_only = head.forward_bev(image_features, None, metas)
    lidar_only = head.forward_bev(None, lidar_features, metas)

    assert image_bev.shape == (2, 16, 4, 3)
    assert lidar_bev.shape == (2, 16, 4, 3)
    assert torch.allclose(image_only, image_bev)
    assert torch.allclose(lidar_only, lidar_bev)


def test_lane_query_decoder_exposes_each_intermediate_layer_for_auxiliary_loss():
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
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
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=2, num_heads=2,
        feedforward_channels=16, dropout=0.0)
    residual = 0.4
    with torch.no_grad():
        decoder.x_refine_heads[0].bias.fill_(residual)

    predictions = decoder(
        torch.randn(2, 8, 2, 2), torch.nn.Embedding(4, 8))
    previous = predictions['aux_predictions'][0]['curve_x_controls']
    normalized_previous = ((previous + 25.0) / 50.0).clamp(1e-4, 1 - 1e-4)
    expected = torch.sigmoid(torch.logit(normalized_previous) + residual) * 50.0 - 25.0

    assert torch.allclose(predictions['curve_x_controls'], expected, atol=1e-5)
    assert not torch.allclose(predictions['curve_x_controls'], previous)


def test_lane_query_decoder_predicts_an_ordered_visibility_interval():
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
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


def test_lane_query_decoder_initializes_objectness_with_a_low_prior():
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.0,
        objectness_prior_probability=0.01)

    assert torch.allclose(
        decoder.cls_head.bias.sigmoid(), torch.tensor([0.01]), atol=1e-6)


def test_lane_query_decoder_adds_denoising_queries_only_during_training():
    decoder = LaneFormerLaneQueryDecoder(
        embed_dims=8, bev_h=2, bev_w=2,
        pc_range=[-25., 0., -5., 25., 100., 5.], num_query=3,
        anchor_y_steps=[0.0, 2.0, 4.0], num_layers=1, num_heads=2,
        feedforward_channels=16, dropout=0.0, dn_num_groups=1)
    bev = torch.randn(2, 8, 2, 2)
    bev_pos = torch.nn.Embedding(4, 8)
    gt_lanes = torch.zeros(2, 2, 3, 3)
    gt_lanes[:, :, :, 1] = torch.tensor([0.0, 2.0, 4.0])
    gt_lanes[:, 0, :, 0] = -2.0
    gt_visibility = torch.zeros(2, 2, 3)
    gt_visibility[:, 0] = 1.0

    train_predictions = decoder(
        bev, bev_pos, dn_lanes_3d=gt_lanes, dn_visibility=gt_visibility)
    inference_predictions = decoder(bev, bev_pos)

    assert train_predictions['pred_x'].shape == (2, 3, 3)
    assert train_predictions['dn_predictions']['pred_x'].shape == (2, 2, 3)
    assert train_predictions['dn_valid_mask'].tolist() == [[True, False], [True, False]]
    assert 'dn_predictions' not in inference_predictions


def test_lane_head_uses_51_anchors_and_backpropagates():
    torch.manual_seed(0)
    anchors = [2.0 * index for index in range(51)]
    head = _head(32, 8, 5, 50, anchors, 16, dn_num_groups=1)
    batch_size = 2
    image_features = [torch.randn(batch_size, 1, 32, 16, 16)]
    lidar_features = [torch.randn(batch_size, 16, 4, 4)]
    metas = _metas(batch_size)
    history = [dict(
        bev=torch.randn(batch_size, 32, 8, 5), img_metas=_metas(batch_size))]

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
    memory_mask = torch.zeros_like(gt_visibility)
    memory_mask[:, 0, 30:35] = 1.0

    losses = head.loss(
        image_features, lidar_features, metas, prev_bev=history,
        gt_lanes_3d=gt_lanes, gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category,
        gt_lanes_3d_enhanced=enhanced_lanes,
        gt_lanes_visibility_enhanced=enhanced_visibility,
        gt_lanes_memory_mask=memory_mask)
    total_loss = sum(losses.values())
    total_loss.backward()

    assert head.anchor_y_steps.numel() == 51
    final_lane_loss_names = {
        'loss_geometry', 'loss_visibility', 'loss_objectness',
        'loss_category', 'loss_visibility_dice',
        'loss_visibility_boundary', 'loss_dn_geometry',
        'loss_dn_visibility', 'loss_dn_visibility_dice',
        'loss_dn_visibility_boundary', 'loss_dn_objectness',
        'loss_dn_category'}
    auxiliary_loss_names = {
        f'loss_aux_0_{name.removeprefix("loss_")}'
        for name in final_lane_loss_names}
    lidar_loss_names = {'loss_lidar_curvature', 'loss_lidar_grade'}
    assert set(losses) == (
        final_lane_loss_names | auxiliary_loss_names | lidar_loss_names)
    assert torch.isfinite(total_loss)
    assert head.lane_decoder.query.weight.grad is not None
    assert torch.isfinite(head.lane_decoder.query.weight.grad).all()
    assert head.anchor_prior.geometry_encoder[-1].weight.grad is not None
    assert torch.isfinite(
        head.anchor_prior.geometry_encoder[-1].weight.grad).all()


def test_hungarian_assignment_is_one_to_one():
    head = _head(16, 2, 2, 2, [0.0, 2.0], 16)
    pred_x = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    pred_z = torch.zeros_like(pred_x)
    pred_vis = torch.full_like(pred_x, 8.0)
    gt_x = torch.tensor([[9.9, 10.1], [0.1, -0.1]])
    gt_z = torch.zeros_like(gt_x)
    gt_vis = torch.ones_like(gt_x)

    pred_objectness = torch.zeros(2)
    pairs = head.loss_module._match(pred_x, pred_z, pred_vis, pred_objectness,
                                    gt_x, gt_z, gt_vis)
    assert pairs == [(0, 1), (1, 0)]


def test_hungarian_assignment_uses_objectness_to_stabilize_close_candidates():
    loss_module = LaneFormerHungarianLaneLoss(
        match_cost_geometry=1.0, match_cost_vis=0.0,
        match_cost_objectness=1.0)
    pred_x = torch.tensor([[0.20, 0.20], [0.25, 0.25]])
    pred_z = torch.zeros_like(pred_x)
    pred_vis = torch.full_like(pred_x, 8.0)
    pred_objectness = torch.tensor([-5.0, 5.0])
    gt_x = torch.zeros(1, 2)
    gt_z = torch.zeros_like(gt_x)
    gt_vis = torch.ones_like(gt_x)

    pairs = loss_module._match(
        pred_x, pred_z, pred_vis, pred_objectness, gt_x, gt_z, gt_vis)

    assert pairs == [(1, 0)]


def test_hungarian_assignment_uses_visibility_dice_overlap():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=0.0, visibility_loss_weight=0.0,
        visibility_dice_loss_weight=0.0,
        visibility_boundary_loss_weight=0.0,
        objectness_loss_weight=1.0, category_loss_weight=0.0,
        match_cost_geometry=0.0, match_cost_vis=0.0,
        match_cost_vis_dice=1.0, match_cost_objectness=0.0)
    pred_objectness = torch.zeros(1, 2, requires_grad=True)
    predictions = dict(
        pred_x=torch.zeros(1, 2, 4), pred_z=torch.zeros(1, 2, 4),
        pred_vis=torch.tensor([[[8.0, 8.0, 8.0, 8.0],
                                [-8.0, 8.0, 8.0, -8.0]]]),
        pred_cls=pred_objectness,
        pred_category=torch.zeros(1, 2, 21))
    gt_lanes = torch.zeros(1, 1, 4, 3)
    gt_visibility = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]])

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility)
    sum(losses.values()).backward()

    assert pred_objectness.grad[0, 0] > 0
    assert pred_objectness.grad[0, 1] < 0


def test_objectness_positive_gradient_is_not_diluted_by_query_count():
    loss_module = LaneFormerHungarianLaneLoss(
        focal_alpha=0.25, focal_gamma=2.0)

    def positive_gradient(num_queries):
        logits = torch.zeros(num_queries, requires_grad=True)
        targets = torch.zeros(num_queries)
        targets[0] = 1.0
        loss_module._focal_loss(logits, targets).backward()
        return logits.grad[0]

    assert torch.allclose(
        positive_gradient(4), positive_gradient(40), atol=1e-7)


def test_hungarian_loss_supervises_each_auxiliary_decoder_layer():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=1.0, visibility_loss_weight=1.0,
        objectness_loss_weight=1.0, category_loss_weight=1.0,
        aux_loss_weight=0.5)

    def prediction(x):
        return dict(
            pred_x=torch.full((1, 1, 2), x, requires_grad=True),
            pred_z=torch.zeros(1, 1, 2, requires_grad=True),
            pred_vis=torch.zeros(1, 1, 2, requires_grad=True),
            pred_cls=torch.zeros(1, 1, requires_grad=True),
            pred_category=torch.zeros(1, 1, 21, requires_grad=True))

    auxiliary = prediction(1.0)
    predictions = prediction(0.5)
    predictions['aux_predictions'] = [auxiliary]
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.ones(1, 1, 2)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=torch.tensor([[2]]))
    sum(losses.values()).backward()

    assert 'loss_aux_0_geometry' in losses
    assert 'loss_aux_0_objectness' in losses
    assert auxiliary['pred_x'].grad is not None
    assert auxiliary['pred_x'].grad.abs().sum() > 0


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


def test_loss_uses_matched_balanced_visibility_and_3d_huber_geometry():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=1.0, visibility_loss_weight=1.0,
        objectness_loss_weight=1.0, category_loss_weight=1.0,
        geometry_huber_beta=0.5)
    predictions = dict(
        # Query 0 is matched to the only GT lane: every sampled point is 5 m away.
        pred_x=torch.tensor([[[3.0, 3.0], [10.0, 10.0]]], requires_grad=True),
        pred_z=torch.tensor([[[4.0, 4.0], [10.0, 10.0]]], requires_grad=True),
        pred_vis=torch.zeros(1, 2, 2, requires_grad=True),
        pred_cls=torch.zeros(1, 2, requires_grad=True),
        pred_category=torch.zeros(1, 2, 21, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.tensor([[[1.0, 0.0]]])
    gt_category = torch.tensor([[2]])

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category)
    total = sum(losses.values())
    total.backward()

    # Smooth L1 with beta=.5 applied to d=sqrt(dx^2+dz^2)=5 gives 4.75.
    assert torch.allclose(losses['loss_geometry'], torch.tensor(4.75))
    # Matched query has one visible and one invisible anchor: equal BCE terms.
    assert torch.allclose(losses['loss_visibility'], torch.log(torch.tensor(2.0)))
    # The unmatched query receives objectness supervision, but no visibility/category.
    assert predictions['pred_cls'].grad[0, 1].abs() > 0
    assert torch.all(predictions['pred_vis'].grad[0, 1] == 0)
    assert torch.all(predictions['pred_category'].grad[0, 1] == 0)


def test_geometry_normalization_preserves_metric_huber_shape_at_smaller_scale():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=1.0, visibility_loss_weight=0.0,
        objectness_loss_weight=0.0, category_loss_weight=0.0,
        geometry_huber_beta=0.5, geometry_normalization=10.0)
    predictions = dict(
        pred_x=torch.tensor([[[3.0, 3.0]]], requires_grad=True),
        pred_z=torch.tensor([[[4.0, 4.0]]], requires_grad=True),
        pred_vis=torch.zeros(1, 1, 2, requires_grad=True),
        pred_cls=torch.zeros(1, 1, requires_grad=True),
        pred_category=torch.zeros(1, 1, 21, requires_grad=True))
    gt_lanes = torch.zeros(1, 1, 2, 3)
    gt_visibility = torch.ones(1, 1, 2)

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility)
    losses['loss_geometry'].backward()

    # smooth_l1(distance=5m, beta=.5m) / 10 = 4.75 / 10.
    assert torch.allclose(losses['loss_geometry'], torch.tensor(0.475))
    assert torch.allclose(
        predictions['pred_x'].grad,
        torch.full_like(predictions['pred_x'], 0.03))


def test_visibility_dice_and_boundary_losses_reward_the_correct_interval():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=0.0, visibility_loss_weight=0.0,
        visibility_dice_loss_weight=1.0,
        visibility_boundary_loss_weight=1.0,
        objectness_loss_weight=0.0, category_loss_weight=0.0)
    gt_lanes = torch.zeros(1, 1, 4, 3)
    gt_visibility = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]])

    def prediction(logits, start, end):
        return dict(
            pred_x=torch.zeros(1, 1, 4), pred_z=torch.zeros(1, 1, 4),
            pred_vis=torch.tensor([[logits]], dtype=torch.float32),
            pred_vis_start=torch.tensor([[start]], dtype=torch.float32),
            pred_vis_end=torch.tensor([[end]], dtype=torch.float32),
            pred_cls=torch.zeros(1, 1),
            pred_category=torch.zeros(1, 1, 21))

    correct = loss_module(
        prediction([-8.0, 8.0, 8.0, -8.0], 1.0, 2.0),
        gt_lanes_3d=gt_lanes, gt_lanes_visibility=gt_visibility)
    wrong = loss_module(
        prediction([8.0, -8.0, -8.0, 8.0], 0.0, 3.0),
        gt_lanes_3d=gt_lanes, gt_lanes_visibility=gt_visibility)

    assert correct['loss_visibility_dice'] < wrong['loss_visibility_dice']
    assert correct['loss_visibility_boundary'] < wrong['loss_visibility_boundary']


def test_temporal_enhanced_lane_uses_its_enhanced_category():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=0.0, visibility_loss_weight=0.0,
        objectness_loss_weight=0.0, category_loss_weight=1.0)
    category_logits = torch.full((1, 1, 21), -10.0)
    category_logits[0, 0, 7] = 10.0
    predictions = dict(
        pred_x=torch.zeros(1, 1, 2), pred_z=torch.zeros(1, 1, 2),
        pred_vis=torch.ones(1, 1, 2), pred_cls=torch.zeros(1, 1),
        pred_category=category_logits)
    lanes = torch.zeros(1, 1, 2, 3)
    visibility = torch.ones(1, 1, 2)

    losses = loss_module(
        predictions, gt_lanes_3d=lanes,
        gt_lanes_visibility=visibility,
        gt_lanes_category=torch.tensor([[2]]),
        gt_lanes_3d_enhanced=lanes,
        gt_lanes_visibility_enhanced=visibility,
        gt_lanes_category_enhanced=torch.tensor([[7]]))

    assert losses['loss_category'] < 1e-6


def test_denoising_queries_receive_direct_lane_supervision():
    loss_module = LaneFormerHungarianLaneLoss(
        geometry_loss_weight=1.0, visibility_loss_weight=1.0,
        objectness_loss_weight=1.0, category_loss_weight=1.0,
        dn_loss_weight=1.0, geometry_huber_beta=0.5)
    predictions = dict(
        pred_x=torch.zeros(1, 1, 2, requires_grad=True),
        pred_z=torch.zeros(1, 1, 2, requires_grad=True),
        pred_vis=torch.zeros(1, 1, 2, requires_grad=True),
        pred_cls=torch.zeros(1, 1, requires_grad=True),
        pred_category=torch.zeros(1, 1, 21, requires_grad=True),
        dn_predictions=dict(
            pred_x=torch.tensor([[[0.1, 0.1], [0.0, 0.0]]], requires_grad=True),
            pred_z=torch.zeros(1, 2, 2, requires_grad=True),
            pred_vis=torch.zeros(1, 2, 2, requires_grad=True),
            pred_cls=torch.zeros(1, 2, requires_grad=True),
            pred_category=torch.zeros(1, 2, 21, requires_grad=True)),
        dn_valid_mask=torch.tensor([[True, False]]),
        dn_target_indices=torch.tensor([0, 1]))
    gt_lanes = torch.zeros(1, 2, 2, 3)
    gt_visibility = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    gt_category = torch.tensor([[2, 0]])

    losses = loss_module(
        predictions, gt_lanes_3d=gt_lanes,
        gt_lanes_visibility=gt_visibility,
        gt_lanes_category=gt_category)
    sum(losses.values()).backward()

    assert set(losses).issuperset({
        'loss_dn_geometry', 'loss_dn_visibility', 'loss_dn_objectness',
        'loss_dn_category'})
    assert losses['loss_dn_geometry'] > 0
    assert predictions['dn_predictions']['pred_x'].grad is not None
    assert predictions['dn_predictions']['pred_cls'].grad is not None
    assert predictions['dn_predictions']['pred_cls'].grad[0, 0] < 0
    assert predictions['dn_predictions']['pred_cls'].grad[0, 1] > 0
