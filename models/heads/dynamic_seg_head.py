"""Dynamic-width segmentation head (DA or Lane) over compact Z.

At width w the hidden layers use the first ceil(w*C) channels (nested prefix);
the final 1x1 conv keeps its full 2-channel output so the eval pipeline
(argmax over 2 channels) is independent of the allocated budget.
"""
import torch
import torch.nn as nn

from models.router.dynamic_conv import DynamicConv2d, DynamicBatchNorm2d


class DynamicSegHead(nn.Module):
    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.conv1 = DynamicConv2d(in_channels, hidden, 3, bias=False)
        self.bn1 = DynamicBatchNorm2d(hidden)
        self.conv2 = DynamicConv2d(hidden, hidden, 3, bias=False)
        self.bn2 = DynamicBatchNorm2d(hidden)
        self.out = DynamicConv2d(hidden, 2, 1, slice_out=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, z, target_hw=None):
        x = self.act(self.bn1(self.conv1(z)))
        x = self.act(self.bn2(self.conv2(x)))
        logits = self.out(x)
        if target_hw is not None:
            logits = nn.functional.interpolate(logits, size=target_hw,
                                               mode="bilinear", align_corners=False)
        return logits
