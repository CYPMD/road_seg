"""U-Net with four downsampling stages, skip connections and raw output logits.

The base width defaults to 32; use 64 for the wider original channel schedule.
This is a from-scratch baseline, without a pretrained encoder.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("Channel counts must be positive")
        widths = [base_channels * (2**i) for i in range(5)]
        self.encoder = nn.ModuleList([DoubleConv(in_channels, widths[0])])
        self.encoder.extend(DoubleConv(a, b) for a, b in zip(widths, widths[1:]))
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.ModuleList(Up(widths[i], widths[i - 1]) for i in range(4, 0, -1))
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        if min(x.shape[-2:]) < 32:
            raise ValueError("U-Net requires image dimensions of at least 32")
        skips = []
        for block in self.encoder[:-1]:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoder[-1](x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        return self.head(x)  # Do not apply sigmoid; the shared pipeline owns it.


def build_model(in_channels=3, out_channels=1, base_channels=32) -> nn.Module:
    return UNet(in_channels, out_channels, base_channels)
