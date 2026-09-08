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
import torch.nn.functional as F

# Must match models/heads/det_head.py defaults (R0) and the anchors fed to YOLOLoss.
DEFAULT_ANCHORS_3S = [[[4, 12], [7, 19], [11, 28]],
                      [[17, 40], [25, 58], [38, 89]],
                      [[62, 136], [88, 206], [124, 412]]]


class _DWResBlock(nn.Module):
    """Depthwise-separable residual block used by the reconstruction variants.

    depthwise 3x3 (groups=c) -> BN -> ReLU -> 1x1 -> BN, residual add, ReLU.
    The dilation rate is the ONLY thing that differs between the dilated arm and
    its control, so any measured difference is attributable to receptive field
    and not to depth, width or parameter count.
    """

    def __init__(self, c, rate):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=rate, dilation=rate, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return F.relu(x + self.f(x), inplace=True)

    def extra_repr(self):
        return "dilation=%d" % self.f[0].dilation[0]


class DetFromZ(nn.Module):
    """3-scale YOLO detection head whose ONLY information source is Compact Z.

    Args:
        zc:     channels of Compact Z.
        nc:     number of classes (1 for BDD100K vehicle).
        proj:   if True, insert a 1x1 conv+BN+ReLU before the pyramid (R2).
                if False, Z feeds the pyramid directly (R1).
        det_ch: fixed internal width of the downsampling branches (see module doc).
        anchors: list of 3 anchor-pair lists, in input pixels (input is 640x640).
        rec: reconstruction applied after the entry projection, before the
            pyramid. "1x1" (default, unchanged behaviour), "dw_plain" (N
            depthwise residual blocks at dilation 1) or "dw_dilated" (same
            blocks at rec_rates). Motivated by YOLOF: a single-level feature has
            a fixed receptive field, and a 1x1 projector has a receptive field of
            exactly one pixel.
    """

    def __init__(self, zc, nc=1, proj=True, det_ch=32, anchors=None, num_anchors=3,
                 rec="1x1", rec_blocks=2, rec_rates=(2, 4),
                 p2="none", p2_ch=None):
        super().__init__()
        self.zc = zc
        self.nc = nc
        self.no = nc + 5
        self.na = num_anchors
        # Phase 4B-3: p2 != "none" prepends a stride-4 level.
        self.p2 = p2
        self.nl = 4 if p2 != "none" else 3
        self.det_ch = det_ch

        if anchors is None:
            anchors = DEFAULT_ANCHORS_3S
        assert len(anchors) == self.nl, (
            f"{self.nl} detection levels need {self.nl} anchor groups, "
            f"got {len(anchors)}")
        self.anchors = torch.tensor(anchors).float().view(self.nl, self.na, 2)
        self.anchor_grid = self.anchors.clone().view(self.nl, 1, self.na, 1, 1, 2)
        # Z is at 1/8 => pyramid strides 8/16/32; with p2 the stride-4
        # level is prepended, so it must come first in every list.
        self.stride = torch.tensor([4, 8, 16, 32] if self.nl == 4
                                   else [8, 16, 32])
        if p2 == "f1":
            assert p2_ch is not None, "p2=\"f1\" needs p2_ch"
            self.lat = nn.Conv2d(p2_ch, det_ch, 1, bias=False)

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

        # --- reconstruction (default "1x1" => Identity => identical to before) ---
        self.rec = rec
        self.rec_blocks = int(rec_blocks)
        self.rec_rates = tuple(rec_rates)
        if rec == "1x1":
            self.rec_op = nn.Identity()
        elif rec in ("dw_plain", "dw_dilated"):
            rates = ([1] * self.rec_blocks if rec == "dw_plain"
                     else list(self.rec_rates)[: self.rec_blocks])
            self.rec_op = nn.Sequential(
                *[_DWResBlock(det_ch, r) for r in rates])
        else:
            raise ValueError("unknown detection rec: %r" % (rec,))

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

    def forward(self, z, hw=None, extra=None):
        """z: (B, zc, H/8, W/8). Returns the same formats as R0's DetectHead.

        extra: the encoder's {"f0": 1/2, "f1": 1/4} dict, required only
        when p2 == "f1" (the native 1/4 lateral).
        """
        x0 = self.rec_op(self.proj(z))   # 1/8, after the reconstruction
        x1 = self.down1(x0)        # 1/16
        x2 = self.down2(x1)        # 1/32
        feats = [x0, x1, x2]
        if self.p2 != "none":
            # Bilinear upsample carries NO new information: it is the
            # grid-resolution arm, the exact parallel of the lane probe's
            # l14up cell. "f1" additionally adds the encoder's real 1/4 map.
            p = F.interpolate(x0, scale_factor=2, mode="bilinear",
                              align_corners=False)
            if self.p2 == "f1":
                f1 = extra["f1"]
                if p.shape[-2:] != f1.shape[-2:]:
                    p = F.interpolate(p, size=f1.shape[-2:], mode="bilinear",
                                      align_corners=False)
                p = p + self.lat(f1)
            feats = [p] + feats

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
