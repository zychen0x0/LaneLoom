import torch

from LaneFormer.models.ground_prior.groundgrid_road_prior import (
    GroundGridRoadPriorExtractor,
)


def _curved_road_points():
    y = torch.arange(2.0, 42.0, 1.0)
    lateral = torch.linspace(-3.0, 3.0, 13)
    yy, offset = torch.meshgrid(y, lateral, indexing='ij')
    center_x = 0.005 * yy.square()
    ground = torch.stack((
        center_x + offset,
        yy,
        -1.5 + 0.002 * yy.square()), dim=-1).reshape(-1, 3)
    obstacle = torch.stack((
        0.005 * y[12:20].square(), y[12:20],
        torch.full_like(y[12:20], 0.5)), dim=-1)
    return torch.cat((ground, obstacle), dim=0), ground.size(0)


def test_groundgrid_extracts_ground_and_curved_road_geometry():
    points, ground_count = _curved_road_points()
    extractor = GroundGridRoadPriorExtractor(
        pc_range=[-20.0, 0.0, -4.0, 20.0, 50.0, 3.0],
        grid_resolution=1.0,
        anchor_y_steps=torch.arange(2.0, 42.0, 2.0).tolist(),
        max_road_hypotheses=2,
        min_ground_points=2,
        ground_height_tolerance=0.3)

    result = extractor([points])

    ground_mask = result['ground_masks'][0]
    assert ground_mask[:ground_count].float().mean() > 0.9
    assert not ground_mask[ground_count:].any()
    assert result['road_valid'][0, 0]
    assert torch.allclose(
        result['road_samples'][0, 0, :, 0],
        0.005 * result['anchor_y_steps'].square(), atol=0.8)
    assert torch.allclose(
        result['road_samples'][0, 0, :, 2],
        -1.5 + 0.002 * result['anchor_y_steps'].square(), atol=0.25)
    middle = slice(3, -3)
    assert torch.allclose(
        result['horizontal_curvature'][0, 0, middle].median(),
        torch.tensor(0.01), atol=0.004)
    assert torch.allclose(
        result['vertical_curvature'][0, 0, middle].median(),
        torch.tensor(0.004), atol=0.002)


def test_groundgrid_keeps_multiple_direction_hypotheses_at_a_fork():
    y = torch.arange(8.0, 48.0)
    lateral = torch.linspace(-2.0, 2.0, 9)
    yy, offset = torch.meshgrid(y, lateral, indexing='ij')
    left = torch.stack((0.18 * (yy - 8) + offset, yy,
                        -1.4 + 0.001 * yy.square()), dim=-1).reshape(-1, 3)
    right = torch.stack((-0.18 * (yy - 8) + offset, yy,
                         -1.4 + 0.001 * yy.square()), dim=-1).reshape(-1, 3)
    points = torch.cat((left, right), dim=0)
    extractor = GroundGridRoadPriorExtractor(
        pc_range=[-20.0, 0.0, -4.0, 20.0, 50.0, 3.0],
        grid_resolution=1.0,
        anchor_y_steps=torch.arange(8.0, 48.0, 2.0).tolist(),
        max_road_hypotheses=3,
        min_ground_points=2,
        ground_height_tolerance=0.3)

    result = extractor([points])

    directions = result['road_directions'][0][result['road_valid'][0]]
    assert directions.numel() >= 2
    assert directions.min() < -0.1
    assert directions.max() > 0.1


def test_groundgrid_rejects_a_dense_elevated_obstacle_patch():
    y = torch.arange(5., 16.)
    road_x = torch.cat((torch.arange(-5., -1.5, .5),
                        torch.arange(2., 5.5, .5)))
    yy, xx = torch.meshgrid(y, road_x, indexing='ij')
    road = torch.stack(
        (xx, yy, torch.full_like(xx, -1.)), dim=-1).reshape(-1, 3)
    road = road.repeat_interleave(2, dim=0)
    obstacle_x = torch.arange(-1., 1.5, .25)
    obstacle_y, obstacle_x = torch.meshgrid(y, obstacle_x, indexing='ij')
    obstacle = torch.stack((
        obstacle_x, obstacle_y, torch.full_like(obstacle_x, 1.)),
        dim=-1).reshape(-1, 3)
    points = torch.cat((road, obstacle), dim=0)
    extractor = GroundGridRoadPriorExtractor(
        pc_range=[-10., 0., -3., 10., 20., 3.],
        grid_resolution=.5,
        anchor_y_steps=[2., 4., 6., 8., 10., 12., 14., 16.],
        max_road_hypotheses=2,
        min_ground_points=2,
        ground_height_tolerance=.25)

    result = extractor([points])

    mask = result['ground_masks'][0]
    assert mask[:road.size(0)].float().mean() > .9
    assert not mask[road.size(0):].any()
