"""Geometry compatibility and spatial evidence retention, without target data."""
import torch
import torch.nn.functional as F

from LaneLoom.models.head.spatial_bev_shape_prior import SpatialBEVShapePrior
from LaneLoom.models.head.shape_proposal_expander import ShapeProposalExpander


def prior():
    return SpatialBEVShapePrior(lidar_in_channels=4, hidden_channels=16,
                                anchor_y_steps=[5., 20., 50., 100.],
                                max_templates=5, longitudinal_rows=8,
                                spatial_layers=1, num_heads=4)


def test_spatial_prior_keeps_five_cubic_templates_and_all_seed_translations():
    torch.manual_seed(7)
    model = prior()
    inputs = dict(features=torch.randn(2, 4, 16, 24),
                  occupancy=torch.ones(2, 1, 16, 24, dtype=torch.bool))
    result = model(inputs)
    assert result['shape_x_controls'].shape == (2, 5, 4)
    assert result['shape_z_controls'].shape == (2, 5, 4)
    assert torch.count_nonzero(result['shape_x_controls'][..., 0]) == 0
    assert torch.count_nonzero(result['shape_z_controls'][..., 0]) == 0
    result['shape_points'].square().mean().backward()
    assert model.feature_encoder[0].weight.grad.abs().sum() > 0
    assert model.row_attention.in_proj_weight.grad.abs().sum() > 0
    expand = ShapeProposalExpander(pc_range=[-25, 0, -5, 25, 100, 5],
                                   anchor_y_steps=[5., 20., 50., 100.],
                                   num_x_seeds=101, template_feature_channels=16, embed_dims=16)
    proposals = expand(result)
    assert proposals['curve_x_controls'].shape == (2, 505, 4)
    expected = result['shape_x_controls'][:, :, None] + expand.x_seeds[None, None, :, None]
    torch.testing.assert_close(proposals['curve_x_controls'], expected.reshape(2, 505, 4))


def test_equal_coarse_averages_do_not_erase_lateral_layout():
    torch.manual_seed(3)
    model = prior().eval()
    left = torch.zeros(1, 4, 16, 16)
    left[..., ::4] = 1
    right = left.roll(2, dims=-1)
    torch.testing.assert_close(F.adaptive_avg_pool2d(left, (4, 4)),
                               F.adaptive_avg_pool2d(right, (4, 4)))
    with torch.no_grad():
        a = model(dict(features=left))['shape_points']
        b = model(dict(features=right))['shape_points']
    assert (a - b).abs().max() > 1e-5


def test_empty_rows_and_empty_clouds_are_finite_and_flagged():
    model = prior()
    occupancy = torch.zeros(2, 1, 16, 24, dtype=torch.bool)
    occupancy[0, :, :4, 10:12] = True
    result = model(dict(features=torch.randn(2, 4, 16, 24), occupancy=occupancy))
    assert result['from_lidar'].tolist() == [True, False]
    assert torch.isfinite(result['shape_points']).all()
    assert torch.count_nonzero(result['shape_x_controls'][1]) == 0
    assert torch.count_nonzero(result['shape_z_controls'][1]) == 0
