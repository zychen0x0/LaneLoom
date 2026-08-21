"""Fuse an image feature pyramid into one high-resolution feature map."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class DilatedMultiScaleFusion(nn.Module):
    """Merge equal-width pyramid levels at the finest input resolution."""

    def __init__(self, in_channels, out_channels, num_scales,
                 dilations=(1, 2, 5)):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_scales = int(num_scales)
        self.dilations = tuple(int(dilation) for dilation in dilations)
        if self.num_scales <= 0:
            raise ValueError('num_scales must be positive.')
        if len(self.dilations) != self.num_scales:
            raise ValueError(
                'dilations must contain one entry per feature scale.')

        self.scale_layers = nn.ModuleList()
        for scale_index in range(self.num_scales):
            layers = []
            scale_dilations = self.dilations[:self.num_scales - scale_index]
            for layer_index, dilation in enumerate(scale_dilations):
                input_channels = (
                    self.in_channels if layer_index == 0
                    else self.out_channels)
                layers.extend((
                    nn.Conv2d(
                        input_channels,
                        self.out_channels,
                        kernel_size=1 if dilation == 1 else 3,
                        padding=0 if dilation == 1 else dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(inplace=True),
                ))
            self.scale_layers.append(nn.Sequential(*layers))

        self.output_layer = nn.Sequential(
            nn.Conv2d(
                self.out_channels, self.out_channels, 3,
                padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 1),
        )

    def forward(self, features):
        if len(features) != self.num_scales:
            raise ValueError(
                f'Expected {self.num_scales} image feature scales, got '
                f'{len(features)}.')

        target_size = features[0].shape[-2:]
        fused_levels = []
        for scale_index, (feature, layers) in enumerate(zip(
                features, self.scale_layers)):
            if feature.size(1) != self.in_channels:
                raise ValueError(
                    f'Image feature scale {scale_index} has '
                    f'{feature.size(1)} channels; expected '
                    f'{self.in_channels}.')
            level = layers(feature)
            if level.shape[-2:] != target_size:
                level = F.interpolate(
                    level, target_size, mode='bilinear', align_corners=True)
            fused_levels.append(level)

        return self.output_layer(torch.stack(fused_levels, dim=0).sum(dim=0))
