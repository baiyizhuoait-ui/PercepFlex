"""Segmentation head (DA or Lane) consuming the compact representation Z.

Small decoder: two depthwise-separable convs on Z (1/8) -> 1x1 conv -> 2-channel
logits at 1/8, bilinear-upsampled to input resolution (640x640) at the end so
the unified evaluation pipeline (argmax over 2 channels) works unchanged.
"""
import torch
import torch.nn as nn


class SegHead(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.out = nn.Conv2d(hidden, 2, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, z, target_hw=None):
        x = self.act(self.bn1(self.conv1(z)))
        x = self.act(self.bn2(self.conv2(x)))
        logits = self.out(x)  # (B, 2, H/8, W/8)
        if target_hw is not None:
            logits = nn.functional.interpolate(logits, size=target_hw,
                                               mode="bilinear", align_corners=False)
        return logits
