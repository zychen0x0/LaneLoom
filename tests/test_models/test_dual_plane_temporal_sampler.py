import torch

from LaneFormer.models.head.dual_plane_temporal_sampler import (
    LaneFormerDualPlaneTemporalSampler,
)


def _temporal_metas(batch_size, time_steps, image_size):
    projection = torch.tensor([
        [1.0, 0.0, 0.0, image_size / 2],
        [0.0, 1.0, 0.0, image_size / 2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return [dict(
        queue={index: dict(
            lane2img=[projection], img_shape=(image_size, image_size),
            lidaradj2lidarcurr=torch.eye(4))
            for index in range(time_steps)},
        current=dict(
            lane2img=[projection], img_shape=(image_size, image_size),
            lidaradj2lidarcurr=torch.eye(4)))
        for _ in range(batch_size)]


def test_sampler_builds_temporal_image_and_lidar_tokens_per_query():
    sampler = LaneFormerDualPlaneTemporalSampler(
        image_in_channels=8, lidar_in_channels=8, embed_dims=8,
        pc_range=[-2., 0., -2., 2., 4., 2.], num_image_levels=2,
        temporal_kernel_size=3)
    image_features = [
        torch.randn(1, 3, 1, 8, 8, 8, requires_grad=True),
        torch.randn(1, 3, 1, 8, 4, 4, requires_grad=True),
    ]
    lidar_bev = torch.randn(1, 8, 4, 4, requires_grad=True)
    curves = torch.tensor([[[
        [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0],
    ], [
        [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 4.0, 0.0],
    ]]], requires_grad=True)

    output = sampler(
        curves, image_features, lidar_bev,
        _temporal_metas(batch_size=1, time_steps=3, image_size=8))

    assert output['tokens'].shape == (1, 2, 9, 8)
    assert output['key_padding_mask'].shape == (1, 2, 9)
    assert output['image_valid_mask'].shape == (1, 2, 3, 2, 3)
    assert output['lidar_valid_mask'].shape == (1, 2, 3)
    output['tokens'].sum().backward()
    assert curves.grad is not None
    assert lidar_bev.grad is not None
    assert sampler.temporal_pointwise.weight.grad is not None


def test_sampler_uses_history_pose_before_projecting_curves():
    sampler = LaneFormerDualPlaneTemporalSampler(
        image_in_channels=4, lidar_in_channels=4, embed_dims=4,
        pc_range=[-2., 0., -2., 2., 4., 2.], num_image_levels=1,
        temporal_kernel_size=3)
    images = [torch.randn(1, 2, 1, 4, 8, 8)]
    lidar_bev = torch.randn(1, 4, 4, 4)
    curves = torch.tensor([[[[0., 0., 0.], [0., 2., 0.]]]])
    metas = _temporal_metas(1, 2, 8)
    history_to_current = torch.eye(4)
    history_to_current[0, 3] = 100.0
    metas[0]['queue'][0]['lidaradj2lidarcurr'] = history_to_current

    output = sampler(curves, images, lidar_bev, metas)

    assert not output['image_valid_mask'][0, 0, :, 0, 0].any()
    assert output['image_valid_mask'][0, 0, :, 0, 1].any()
