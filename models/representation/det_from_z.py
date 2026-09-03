"""Task 2(c) — R1/R2 variant: Detection consumes Compact Z (3-scale, fair vs R0).

CONTEXT
-------
R0 (default, unchanged): DetHead reads the encoder's multi-scale features
    [F2(1/8), F3(1/16), F4(1/32)] directly, 3 YOLO scales, stride [8, 16, 32].
R1 (from_z, z_proj=False): detection is derived ONLY from Compact Z.
R2 (from_z, z_proj=True):  same, but Z first passes a 1x1 projection (the
    "minimal information-allocation" path we actually want to test).

FAIRNESS REQUIREMENT (why this was rewritten)
---------------------------------------------
A first draft emitted a SINGLE detection scale; YOLOLoss indexes nl = anchors.shape[0]
= 3 scales, so it crashed with IndexError, and — more importantly — a 1-scale head
would lose for "missing scales" rather than for "information bottleneck", which
invalidates the R0-vs-R2 comparison. This version emits the SAME 3 scales, same
strides, same anchors as R0, so the only difference is the INFORMATION SOURCE
(encoder multi-scale features vs. one unified Compact Z).

Compact Z lives at 1/8 resolution (compact_z.py interpolates to f2.shape[2:]).
We therefore build the detection pyramid from Z by 2x/4x downsampling:
    s0 = Z            (1/8,  stride 8)
    s1 = down(Z)      (1/16, stride 16)
    s2 = down(down(Z))(1/32, stride 32)

PARAMETER-ISOLATION DESIGN
--------------------------
The downsampling branches use a FIXED internal width `det_ch` (default 32), NOT
z_channels. If they used z_channels, R2's head params would grow as O(z^2) across
the Z sweep and the "Z capacity" effect would be confounded with "head capacity".
With det_ch fixed, R2's total param count is essentially constant across Z
(only the 1x1 entry projection scales as zc*det_ch), keeping the sweep clean.

Flag-gated by model.detection.from_z (default off) => R0 path is untouched.
"""
import math

import torch
import torch.nn as nn

# Must match models/heads/det_head.py defaults (R0) and the anchors fed to YOLOLoss.
DEFAULT_ANCHORS_3S = [[[4, 12], [7, 19], [11, 28]],
                      [[17, 40], [25, 58], [38, 89]],
                      [[62, 136], [88, 206], [124, 412]]]


class DetFromZ(nn.Module):
    """3-scale YOLO detection head whose ONLY information source is Compact Z.

    Args:
        zc:     channels of Compact Z.
        nc:     number of classes (1 for BDD100K vehicle).
        proj:   if True, insert a 1x1 conv+BN+ReLU before the pyramid (R2).
                if False, Z feeds the pyramid directly (R1).
        det_ch: fixed internal width of the downsampling branches (see module doc).
        anchors: list of 3 anchor-pair lists, in input pixels (input is 640x640).
    """

    def __init__(self, zc, nc=1, proj=True, det_ch=32, anchors=None, num_anchors=3):
        super().__init__()
        self.zc = zc
        self.nc = nc
        self.no = nc + 5
        self.na = num_anchors
        self.nl = 3
        self.det_ch = det_ch

        if anchors is None:
            anchors = DEFAULT_ANCHORS_3S
        self.anchors = torch.tensor(anchors).float().view(self.nl, self.na, 2)
        self.anchor_grid = self.anchors.clone().view(self.nl, 1, self.na, 1, 1, 2)
        # Z is at 1/8 => pyramid strides 8/16/32.
        self.stride = torch.tensor([8, 16, 32])

        # --- entry projection (R2 only; R1 keeps Identity => Z used raw) ---
        if proj:
            self.proj = nn.Sequential(
                nn.Conv2d(zc, det_ch, 1, bias=False),
                nn.BatchNorm2d(det_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(zc, det_ch, 1, bias=False) if zc != det_ch else nn.Identity(),
                nn.BatchNorm2d(det_ch) if zc != det_ch else nn.Identity(),
                nn.Identity(),
            )

        # --- 1/8 -> 1/16 -> 1/32 (fixed width det_ch, independent of zc) ---
        def _down():
            return nn.Sequential(
                nn.Conv2d(det_ch, det_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(det_ch),
                nn.ReLU(inplace=True),
            )

        self.down1 = _down()
        self.down2 = _down()

        # --- per-scale 1x1 prediction convs (same as R0's DetectHead) ---
        self.m = nn.ModuleList(
            nn.Conv2d(det_ch, self.no * self.na, 1) for _ in range(self.nl))
        self.grid = [torch.zeros(1)] * self.nl
        self._init_bias()

    def _init_bias(self):
        # YOLOv5-style bias init, identical to models/heads/det_head.py.
        for mi, m in enumerate(self.m):
            b = m.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / self.stride[mi]) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99))
            m.bias = torch.nn.Parameter(b.view(-1))

    def forward(self, z, hw=None):
        """z: (B, zc, H/8, W/8). Returns the same formats as R0's DetectHead."""
        x0 = self.proj(z)          # 1/8
        x1 = self.down1(x0)        # 1/16
        x2 = self.down2(x1)        # 1/32
        feats = [x0, x1, x2]

        if self.training:
            raw = []
            for i, x in enumerate(feats):
                x = self.m[i](x)
                bs, _, ny, nx = x.shape
                x = x.view(bs, self.na, self.no, ny * nx).permute(0, 1, 3, 2) \
                     .view(bs, self.na, ny, nx, self.no).contiguous()
                raw.append(x)
            return raw  # list of (B, na, ny, nx, no) for YOLOLoss

        outs = []
        for i, x in enumerate(feats):
            x = self.m[i](x)
            bs, _, ny, nx = x.shape
            x = x.view(bs, self.na, self.no, ny * nx).permute(0, 1, 3, 2) \
                 .view(bs, self.na, ny, nx, self.no).contiguous()
            if self.grid[i].shape[2:4] != x.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x.device)
            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].to(x.device)
            outs.append(y.view(bs, -1, self.no))
        return torch.cat(outs, 1)

    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
