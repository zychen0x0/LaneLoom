import torch
import pytest

from LaneFormer.models.backbones.dynamic_pillar_bev import (
    LaneFormerDynamicPillarBEVEncoder,
)


def _encoder():
    encoder = LaneFormerDynamicPillarBEVEncoder(
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

    assert encoder.point_input_channels == 12
    assert encoder.pillar_statistic_channels == 10
    assert pillars['coordinates'].shape == (1, 3)
    assert pillars['features'].shape == (1, encoder.pillar_feature_channels)
    assert pillars['statistics'].shape == (1, 10)


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
