import torch
import torch.nn as nn
import numpy as np


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)


class TurbulenceUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, upscale_factor=16):
        super().__init__()

        # Feature Extraction (Encoder)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        # Super-Resolution Upsampler
        n_upsamples = int(np.log2(upscale_factor))
        upsample_layers = []
        current_channels = 64

        for _ in range(n_upsamples):
            upsample_layers.extend([
                nn.Conv2d(current_channels, current_channels *
                          4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ])

        self.upsampler = nn.Sequential(*upsample_layers)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        x = self.conv_up1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x1], dim=1))

        x = self.upsampler(x)
        return self.outc(x)
