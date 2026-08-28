"""Lightweight multi-scale encoder (Module A).

Design constraints from the taskbook:
- lightweight: depthwise-separable conv + pointwise conv, ESPNet-flavored blocks
  as a *starting point*, but no attention / no large FPN / no exotic modules.
- produces multi-scale compact features (1/8, 1/16, 1/32) consumed by the
  Compact Representation Z and the task heads.

Structure:
    stem (stride 2) -> s1 (stride 2) -> s2 (stride 2) -> s3 (stride 2) -> s4 (stride 2)
    outputs: [F2 (1/8), F3 (1/16), F4 (1/32)]

Channel widths are configurable so the whole model can be compared under equal
parameter budgets (and later serve as the fixed-max-capacity stage for the
Task-Resource Router experiments).
"""
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """DW conv + PW conv, the workhorse of the encoder."""

    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, s, k // 2, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class DWSBlock(nn.Module):
    """Two depthwise-separable convs with residual (strided first conv when s=2)."""

    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(cin, cout, k, s)
        self.conv2 = DepthwiseSeparableConv(cout, cout, k, 1)
        self.shortcut = None
        if s != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, s, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x if self.shortcut is None else self.shortcut(x)
        return self.act(self.conv2(self.conv1(x)) + res)


class LightEncoder(nn.Module):
    """Multi-scale lightweight encoder.

    cfg: {"stem": int, "stages": [int, int, int, int], "blocks": [int, int, int]}
    outputs: list of feature maps [F2 (1/8), F3 (1/16), F4 (1/32)].
    """

    def __init__(self, cfg):
        super().__init__()
        c_stem = cfg.get("stem", 16)
        c1, c2, c3, c4 = cfg.get("stages", [32, 64, 96, 128])
        b2, b3, b4 = cfg.get("blocks", [2, 2, 2])

        self.stem = ConvBNAct(3, c_stem, 3, 2)
        self.s1 = DWSBlock(c_stem, c1, s=2)
        self.s2 = self._make(c1, c2, b2)
        self.s3 = self._make(c2, c3, b3)
        self.s4 = self._make(c3, c4, b4)

    @staticmethod
    def _make(cin, cout, n):
        layers = [DWSBlock(cin, cout, s=2)]
        for _ in range(n - 1):
            layers.append(DWSBlock(cout, cout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)      # 1/2
        x = self.s1(x)        # 1/4
        f2 = self.s2(x)       # 1/8
        f3 = self.s3(f2)      # 1/16
        f4 = self.s4(f3)      # 1/32
        return [f2, f3, f4]
