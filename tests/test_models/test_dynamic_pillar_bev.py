import torch
import pytest

from LaneLoom.models.backbones.dynamic_pillar_bev import (
    DynamicPillarBEVEncoder,
)


def _encoder():
    encoder = DynamicPillarBEVEncoder(
        point_cloud_range=[-1., 0., -2., 1., 2., 2.],
        pillar_size=[.5, .5],
        out_channels=16,
        queue_length=2,
    )
    encoder.eval()
    return encoder


def test_dynamic_pillar_raster_preserves_height_statistics():
    encoder = _encoder()
    points = [torch.tensor([
        [-.75, .25, -1., 2., 4., 0.],
        [-.70, .20, 1., 6., 8., 1.],
    ])]

    raster = encoder.rasterize(points)
    cell = raster[0, :, 0, 0]

    assert raster.shape == (1, 10, 4, 4)
    assert torch.allclose(cell[:5], torch.tensor([
        torch.log(torch.tensor(3.)).item(), -1., 1., 0., 1.
    ]), atol=1e-6)
    expected_intensity = torch.log1p(torch.tensor([2., 6.]))
    assert torch.allclose(
        cell[5:7],
        torch.tensor([
            expected_intensity.mean(), expected_intensity.max()
        ]))
    assert torch.allclose(cell[7:9], torch.tensor([6., 8.]))
    assert torch.allclose(cell[9], torch.tensor(.5))


def test_dynamic_pfe_aggregates_point_features_once_per_pillar():
    encoder = _encoder()
    points = [torch.tensor([
        [-.75, .25, -1., 2., 4., 0.],
        [-.70, .20, 1., 6., 8., 1.],
    ])]

    pillars = encoder.pillarize(points)

    assert encoder.point_input_channels == 15
    assert encoder.pillar_statistic_channels == 10
    assert pillars['coordinates'].shape == (1, 3)
    assert pillars['features'].shape == (1, encoder.pillar_feature_channels)
    assert pillars['statistics'].shape == (1, 10)


def test_dynamic_pillar_paints_points_from_their_temporal_normalized_images():
    encoder = _encoder()
    points = [torch.tensor([
        [1., 0., 1., 1., 0., 0.],
        [0., 1., 1., 1., 0., 1.],
    ])]
    images = torch.tensor([[
        [[
            [[10., 11.], [12., 13.]],
            [[20., 21.], [22., 23.]],
            [[30., 31.], [32., 33.]],
        ]],
        [[
            [[40., 41.], [42., 43.]],
            [[50., 51.], [52., 53.]],
            [[60., 61.], [62., 63.]],
        ]],
    ]])
    projection = torch.tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])
    history_to_current = torch.eye(4)
    history_to_current[0, 3] = 1.
    metas = [dict(queue={
        0: dict(
            lidar2img=[projection],
            lidaradj2lidarcurr=history_to_current),
        1: dict(
            lidar2img=[projection],
            lidaradj2lidarcurr=torch.eye(4)),
    })]

    painted = encoder.paint_points(points, images, metas)

    assert painted[0].shape == (2, 9)
    assert torch.equal(
        painted[0][0],
        torch.tensor([1., 0., 1., 1., 0., 10., 20., 30., 0.]))
    assert torch.equal(
        painted[0][1],
        torch.tensor([0., 1., 1., 1., 0., 41., 51., 61., 1.]))


def test_dynamic_pillar_uses_zero_rgb_for_invalid_image_projections():
    encoder = _encoder()
    points = [torch.tensor([[0., 0., -1., 1., 0., 0.]])]
    images = torch.ones(1, 1, 1, 3, 2, 2)
    metas = [dict(queue={0: dict(
        lidar2img=[torch.eye(4)],
        lidaradj2lidarcurr=torch.eye(4))})]

    painted = encoder.paint_points(points, images, metas)

    assert torch.equal(painted[0][0, 5:8], torch.zeros(3))


def test_dynamic_pillar_loads_pre_painting_point_encoder_weights():
    encoder = _encoder()
    old_weight = torch.arange(
        encoder.pfe_channels * 12, dtype=torch.float32
    ).reshape(encoder.pfe_channels, 12)

    encoder.load_state_dict(
        {'point_mlp.0.weight': old_weight}, strict=False)

    loaded = encoder.point_mlp[0].weight.detach()
    assert torch.equal(loaded[:, :5], old_weight[:, :5])
    assert torch.equal(loaded[:, 5:8], torch.zeros_like(loaded[:, 5:8]))
    assert torch.equal(loaded[:, 8], old_weight[:, 5])
    assert torch.equal(loaded[:, 9:], old_weight[:, 6:])


def test_dynamic_pillar_migrates_rgb_at_tail_point_encoder_weights():
    encoder = _encoder()
    old_weight = torch.arange(
        encoder.pfe_channels * 15, dtype=torch.float32
    ).reshape(encoder.pfe_channels, 15)

    encoder.load_state_dict(
        {'point_mlp.0.weight': old_weight}, strict=False)

    loaded = encoder.point_mlp[0].weight.detach()
    assert torch.equal(loaded[:, :5], old_weight[:, :5])
    assert torch.equal(loaded[:, 5:8], old_weight[:, 12:15])
    assert torch.equal(loaded[:, 8], old_weight[:, 5])
    assert torch.equal(loaded[:, 9:12], old_weight[:, 6:9])
    assert torch.equal(loaded[:, 12:15], old_weight[:, 9:12])


def test_dynamic_pillar_keeps_current_point_encoder_layout_when_loading():
    source = _encoder()
    expected = torch.randn_like(source.point_mlp[0].weight)
    source.point_mlp[0].weight.data.copy_(expected)
    target = _encoder()

    target.load_state_dict(source.state_dict())

    assert torch.equal(target.point_mlp[0].weight, expected)


def test_lane_neck_updates_batch_norm_once_per_training_step():
    encoder = _encoder()
    encoder.train()
    features = torch.randn(2, 16, 4, 4, requires_grad=True)
    batch_norm = encoder.lane_neck.post_conv[1]

    output = encoder.lane_neck(features)
    output.mean().backward()

    assert batch_norm.num_batches_tracked.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='spconv requires CUDA')
def test_dynamic_pillar_returns_stride_two_features_and_empty_occupancy():
    encoder = _encoder()
    encoder = encoder.cuda()
    points = [
        torch.tensor([[-.75, .25, 0., 1., 0., 0.]]),
        torch.empty(0, 6),
    ]
    points = [sample.cuda() for sample in points]

    output = encoder(points)

    assert set(output) == {'features', 'occupancy'}
    assert output['features'].shape == (2, 16, 2, 2)
    assert output['occupancy'].shape == (2, 1, 2, 2)
    assert output['occupancy'].dtype == torch.bool
    assert output['occupancy'][0].any()
    assert not output['occupancy'][1].any()
    assert torch.all(output['features'][1] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='spconv requires CUDA')
def test_configured_sparse_lane_grid_densifies_at_half_pillar_resolution():
    encoder = DynamicPillarBEVEncoder(
        point_cloud_range=[-25., 0., -5., 25., 100., 5.],
        pillar_size=[.25, .25],
        out_channels=128,
        base_channels=32,
        queue_length=3,
    ).cuda().eval()
    points = [torch.tensor(
        [[0., 10., 0., 1., 0., 2.]], device='cuda')]

    output = encoder(points)

    assert output['features'].shape == (1, 128, 200, 100)
    assert output['occupancy'].shape == (1, 1, 200, 100)
