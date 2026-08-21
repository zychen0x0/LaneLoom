import torch

from LaneLoom.models.fusion_layers import DilatedMultiScaleFusion


def test_dilated_multi_scale_fusion_returns_one_high_resolution_level():
    fusion = DilatedMultiScaleFusion(
        in_channels=8,
        out_channels=8,
        num_scales=3,
        dilations=(1, 2, 5),
    ).eval()
    features = [
        torch.randn(2, 8, 16, 24),
        torch.randn(2, 8, 8, 12),
        torch.randn(2, 8, 4, 6),
    ]

    with torch.no_grad():
        fused = fusion(features)

    assert fused.shape == features[0].shape


def test_dilated_multi_scale_fusion_backpropagates_to_every_level():
    fusion = DilatedMultiScaleFusion(
        in_channels=8,
        out_channels=8,
        num_scales=3,
        dilations=(1, 2, 5),
    )
    features = [
        torch.randn(2, 8, 16, 24, requires_grad=True),
        torch.randn(2, 8, 8, 12, requires_grad=True),
        torch.randn(2, 8, 4, 6, requires_grad=True),
    ]

    fusion(features).sum().backward()

    assert all(feature.grad is not None for feature in features)
    assert all(torch.count_nonzero(feature.grad) > 0 for feature in features)
