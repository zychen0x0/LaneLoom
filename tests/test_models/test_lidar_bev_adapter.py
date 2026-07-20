import torch

from LaneLoom.models.fusion_layers.lidar_bev_adapter import (
    LiDARBEVAdapter,
)


def test_lidar_adapter_keeps_lane_channels_until_curve_sampling():
    adapter = LiDARBEVAdapter(
        embed_dims=128, lidar_in_channels=128)
    features = torch.randn(2, 128, 20, 10)

    output = adapter(dict(features=features), output_size=(20, 10))

    assert output is features
