"""YOLO-style detection head (single class by default, matching baselines).

Consumes multi-scale features [F2(1/8), F3(1/16), F4(1/32)] and produces the
same decoded output format as YOLOP/TriLiteNet: (1, na*nx*ny*3, 5+nc) with
[xywh, obj, cls], so the unified evaluation pipeline works unchanged.
"""
import math

import torch
import torch.nn as nn


class DetectHead(nn.Module):
    def __init__(self, channels, nc=1, anchors=None, num_anchors=3):
        """
        channels: [C2, C3, C4] (1/8, 1/16, 1/32)
        anchors: list of 3 anchor-pair lists (one per scale), in input pixels.
        """
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.na = num_anchors
        self.nl = 3
        if anchors is None:
            anchors = [[[4, 12], [7, 19], [11, 28]],
                       [[17, 40], [25, 58], [38, 89]],
                       [[62, 136], [88, 206], [124, 412]]]
        self.anchors = torch.tensor(anchors).float().view(self.nl, self.na, 2)
        self.anchor_grid = self.anchors.clone().view(self.nl, 1, self.na, 1, 1, 2)
        self.stride = torch.tensor([8, 16, 32])
        self.m = nn.ModuleList(
            nn.Conv2d(ch, self.no * self.na, 1) for ch in channels)
        self.grid = [torch.zeros(1)] * self.nl
        self._init_weights()

    def _init_weights(self):
        for mi, m in enumerate(self.m):
            # follow YOLOv5 init: bias for obj=0, cls=0, no-object prior
            b = m.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / self.stride[mi]) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99))
            m.bias = torch.nn.Parameter(b.view(-1))

    def forward(self, xs):
        z = []
        for i, x in enumerate(xs):
            x = self.m[i](x)
            bs, _, ny, nx = x.shape
            x = x.view(bs, self.na, self.no, ny * nx).permute(0, 1, 3, 2) \
                 .view(bs, self.na, ny, nx, self.no).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x.device)
                y = x.sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].to(x.device)
                z.append(y.view(bs, -1, self.no))
            else:
                z.append(x.view(bs, -1, self.no))
        return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
