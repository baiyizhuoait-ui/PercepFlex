"""Nested-prefix dynamic-width conv layers (taskbook §6), optimized.

The width mechanism is contiguous-channel prefix slicing: at width w the layer
uses the FIRST round(w*C) output channels and the first round(w*C_prev) input
channels. Sliced weights/BN stats are CACHED per width when set_width() is
called (once per allocation change), so per-forward overhead is a single
F.conv2d / F.batch_norm call — hardware-friendly and fast.
"""
import math

import torch
import torch.nn as nn


class DynamicConv2d(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=None, g=1, bias=False, slice_out=True):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.k, self.s = k, s
        self.p = p if p is not None else k // 2
        self.g = g
        self.slice_out = slice_out
        self.conv = nn.Conv2d(cin, cout, k, s, self.p, groups=g, bias=bias)
        self.width = 1.0
        self._cache = {}

    def set_width(self, w):
        self.width = w
        if w >= 1.0:
            return
        if w in self._cache:
            return
        n_in = max(1, math.ceil(self.cin * w))
        n_out = max(1, math.ceil(self.cout * w)) if self.slice_out else self.cout
        groups = n_in if self.g == self.cin else self.g
        bias = self.conv.bias[:n_out] if self.conv.bias is not None else None
        self._cache[w] = (self.conv.weight[:n_out, :n_in], bias, n_in, groups)

    def forward(self, x):
        w = self.width
        if w >= 1.0:
            return self.conv(x)
        ww, bias, n_in, groups = self._cache[w]
        return nn.functional.conv2d(
            x[:, :n_in], ww, bias, stride=self.s, padding=self.p, groups=groups)


class DynamicBatchNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.width = 1.0
        self._cache = {}

    def set_width(self, w):
        self.width = w
        if w >= 1.0:
            return
        if w in self._cache:
            return
        n = max(1, int(math.ceil(self.bn.num_features * w)))
        self._cache[w] = (self.bn.running_mean[:n], self.bn.running_var[:n],
                          self.bn.weight[:n], self.bn.bias[:n], n)

    def forward(self, x):
        w = self.width
        full = self.bn.num_features
        if w >= 1.0 and x.shape[1] == full:
            return self.bn(x)
        if w in self._cache:
            rm, rv, wt, bs, n = self._cache[w]
        else:
            n = max(1, int(math.ceil(full * w)))
            rm, rv, wt, bs = (self.bn.running_mean[:n], self.bn.running_var[:n],
                              self.bn.weight[:n], self.bn.bias[:n])
        n = min(n, x.shape[1])
        if self.training:
            return nn.functional.batch_norm(
                x[:, :n], rm[:n], rv[:n], wt[:n], bs[:n], True,
                self.bn.momentum, self.bn.eps)
        # eval: affine form (no BN-stat update), single elementwise kernel
        y = (x[:, :n] - rm[:n].view(1, -1, 1, 1)) * \
            (rv[:n].view(1, -1, 1, 1) + self.bn.eps).rsqrt()
        return y * wt[:n].view(1, -1, 1, 1) + bs[:n].view(1, -1, 1, 1)


def set_width_recursive(module, w):
    """Set width on a module tree, caching sliced weights where supported."""
    for m in module.modules():
        if hasattr(m, "set_width"):
            m.set_width(w)
        elif hasattr(m, "width"):
            m.width = w
