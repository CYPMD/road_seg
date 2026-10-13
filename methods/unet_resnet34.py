"""U-Net with a ResNet-34 encoder and raw, full-resolution output logits.

The encoder uses the standard [3, 4, 6, 3] basic-block layout with widths
64, 128, 256 and 512. It is initialized from scratch; no pretrained weights
or torchvision dependency are required. ``base_channels`` controls only the
decoder width, leaving the ResNet-34 encoder unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .unet import DoubleConv


class _BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class _ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        previous = 64
        for index, (width, count) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3))):
            blocks = [_BasicBlock(previous, width, stride=1 if index == 0 else 2)]
            blocks.extend(_BasicBlock(width, width) for _ in range(count - 1))
            self.stages.append(nn.Sequential(*blocks))
            previous = width
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("Channel counts must be positive")
        self.encoder = _ResNet34Encoder(in_channels)
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


def build_model(in_channels=3, out_channels=1, base_channels=32) -> nn.Module:
    return ResNet34UNet(in_channels, out_channels, base_channels)
