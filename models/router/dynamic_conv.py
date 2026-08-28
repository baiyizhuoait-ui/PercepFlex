"""Nested-prefix dynamic-width conv layers (taskbook §6).

The width mechanism is contiguous-channel prefix slicing: at width w the layer
uses the FIRST round(w*C) output channels and the first round(w*C_prev) input
channels. This is hardware-friendly (TensorRT/PyTorch can execute contiguous
slices) and identical to the static model at w=1.0, so static training IS the
max-capacity stage (Stage A of the taskbook training schedule).
"""
import math

import torch
import torch.nn as nn


def _slice_ch(x, w):
    """First round(w*C) channels of x (B, C, H, W)."""
    if w >= 1.0:
        return x
    c = int(math.ceil(x.shape[1] * w))
    return x[:, :c]


class DynamicConv2d(nn.Module):
    """Conv2d whose active input/output channels follow a nested prefix at width w.

    slice_out=False keeps the full output channels (used for the final head
    layer so the output format never changes with width)."""

    def __init__(self, cin, cout, k=3, s=1, p=None, g=1, bias=False, slice_out=True):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.k, self.s = k, s
        self.p = p if p is not None else k // 2
        self.g = g
        self.slice_out = slice_out
        self.conv = nn.Conv2d(cin, cout, k, s, self.p, groups=g, bias=bias)
        self.width = 1.0  # active width ratio; set by the router / model

    def forward(self, x):
        w = self.width
        if w >= 1.0:
            return self.conv(x)
        n_in = max(1, math.ceil(self.cin * w))
        n_out = max(1, math.ceil(self.cout * w)) if self.slice_out else self.cout
        if self.g == self.cin:  # depthwise: groups follow the sliced input channels
            groups = n_in
        else:
            groups = self.g
        ww = self.conv.weight[:n_out, :n_in]
        return nn.functional.conv2d(
            x[:, :n_in], ww, self.conv.bias[:n_out] if self.conv.bias is not None else None,
            stride=self.s, padding=self.p, groups=groups)


class DynamicBatchNorm2d(nn.Module):
    """BatchNorm2d that normalizes only the active (first n) channels.

    The active channel count is driven by `width` AND by how many channels the
    previous dynamic conv actually produced (x may arrive already sliced).
    """

    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.width = 1.0

    def forward(self, x):
        w = self.width
        full = self.bn.num_features
        n = full if w >= 1.0 else max(1, int(math.ceil(full * w)))
        n = min(n, x.shape[1])
        if n == full and x.shape[1] == full:
            return self.bn(x)
        return nn.functional.batch_norm(
            x[:, :n], self.bn.running_mean[:n], self.bn.running_var[:n],
            self.bn.weight[:n], self.bn.bias[:n], self.bn.training,
            self.bn.momentum, self.bn.eps)


class DWSBlockDyn(nn.Module):
    """Depthwise-separable block with nested-prefix dynamic width (s=1 only)."""

    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.dw = DynamicConv2d(cin, cin, k, 1, groups=cin)
        self.bn1 = DynamicBatchNorm2d(cin)
        self.pw = DynamicConv2d(cin, cout, 1)
        self.bn2 = DynamicBatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        w = self.width = self.dw.width
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw(y)))
        if y.shape[1] < x.shape[1]:
            x = x[:, :y.shape[1]]
        return y + x
