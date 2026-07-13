import torch

from LaneFormer.models.head.laneformer_bev_lane_head import LaneFormerBEVLaneHead
from LaneFormer.models.head.laneformer_lane_query_decoder import (
    LaneFormerLaneQueryDecoder,
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


def _head(embed_dims, bev_h, bev_w, num_query, anchors, lidar_channels):
    return LaneFormerBEVLaneHead(
        bev_encoder=dict(
            type='LaneFormerBEVEncoder', embed_dims=embed_dims,
            bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.],
            spatial_cross_attn=dict(
                type='LaneFormerSpatialCrossAttention', embed_dims=embed_dims,
                bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.],
                num_points_in_pillar=4, num_levels=1),
            lidar_fusion=dict(
                type='LaneFormerImageLiDARFusion', embed_dims=embed_dims,
                lidar_in_channels=lidar_channels, num_layers=1, num_heads=4,
                feedforward_channels=embed_dims * 2, dropout=0.0),
            temporal_self_attn=dict(
                type='LaneFormerTemporalSelfAttention', embed_dims=embed_dims,
                bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.])),
        lane_decoder=dict(
            type='LaneFormerLaneQueryDecoder', embed_dims=embed_dims,
            bev_h=bev_h, bev_w=bev_w, pc_range=[-25., 0., -5., 25., 100., 5.],
            num_query=num_query, anchor_y_steps=anchors, num_layers=2,
            num_heads=4, feedforward_channels=embed_dims * 2, dropout=0.0),
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


def test_lane_head_uses_51_anchors_and_backpropagates():
    torch.manual_seed(0)
    anchors = [2.0 * index for index in range(51)]
    head = _head(32, 8, 5, 40, anchors, 16)
    batch_size = 2
    image_features = [torch.randn(batch_size, 1, 32, 16, 16)]
    lidar_features = [torch.randn(batch_size, 16, 4, 4)]
    metas = _metas(batch_size)
    history = [dict(
        bev=torch.randn(batch_size, 32, 8, 5), img_metas=_metas(batch_size))]

    gt_lanes = torch.zeros(batch_size, 20, 51, 3)
    gt_lanes[:, :, :, 1] = torch.tensor(anchors)
    gt_visibility = torch.zeros(batch_size, 20, 51)
    gt_lanes[:, 0, :, 0] = -2.0
    gt_lanes[:, 0, :, 2] = 0.2
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
    assert set(losses) == {
        'loss_geometry', 'loss_visibility', 'loss_objectness',
        'loss_category'}
    assert torch.isfinite(total_loss)
    assert head.lane_decoder.query.weight.grad is not None
    assert torch.isfinite(head.lane_decoder.query.weight.grad).all()


def test_hungarian_assignment_is_one_to_one():
    head = _head(16, 2, 2, 2, [0.0, 2.0], 16)
    pred_x = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    pred_z = torch.zeros_like(pred_x)
    pred_vis = torch.full_like(pred_x, 8.0)
    gt_x = torch.tensor([[9.9, 10.1], [0.1, -0.1]])
    gt_z = torch.zeros_like(gt_x)
    gt_vis = torch.ones_like(gt_x)

    pairs = head.loss_module._match(pred_x, pred_z, pred_vis,
                                    gt_x, gt_z, gt_vis)
    assert pairs == [(0, 1), (1, 0)]


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
