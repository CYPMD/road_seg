"""U-Net with a torchvision ResNet-34 encoder and full-resolution raw logits.

The encoder uses the standard ResNet-34 [3, 4, 6, 3] layout with widths
64, 128, 256 and 512. Set ``pretrained=True`` to initialize the RGB encoder
from torchvision's ImageNet-1K weights. ``base_channels`` controls only the
decoder width; the ResNet-34 encoder widths remain fixed.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34

from .unet import DoubleConv


class _ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int, pretrained: bool = False):
        super().__init__()
        if pretrained and in_channels != 3:
            raise ValueError("ImageNet-pretrained ResNet-34 requires in_channels=3")

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        # torchvision's pretrained conv1 is RGB-only. For other channel counts,
        # keep the same architecture but create a fresh first convolution.
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.stages = nn.ModuleList([
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ])

    def forward(self, x):
        x = self.stem(x)
        features = [x]
        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features  # Strides 2, 4, 8, 16 and 32.


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, size, skip=None):
        x = self.up(x)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, pretrained=False):
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("Channel counts must be positive")
        self.encoder = _ResNet34Encoder(in_channels, pretrained=pretrained)
        widths = [base_channels * factor for factor in (8, 4, 2, 1, 1)]
        self.decoder = nn.ModuleList()
        previous = 512
        for skip_channels, width in zip((256, 128, 64, 64), widths[:4]):
            self.decoder.append(_Up(previous, skip_channels, width))
            previous = width
        self.final_up = _Up(previous, 0, widths[4])
        self.head = nn.Conv2d(widths[4], out_channels, 1)

    def forward(self, x):
        if min(x.shape[-2:]) < 32:
            raise ValueError("ResNet-34 U-Net requires image dimensions of at least 32")
        size = x.shape[-2:]
        features = self.encoder(x)
        x = features[-1]
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            x = block(x, skip.shape[-2:], skip)
        x = self.final_up(x, size)
        return self.head(x)  # Do not apply sigmoid; the shared pipeline owns it.


def build_model(in_channels=3, out_channels=1, base_channels=32, pretrained=False) -> nn.Module:
    return ResNet34UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        pretrained=pretrained,
    )
