"""Compact Driving Representation Z (Module A, second half).

Light FPN-style fusion of the encoder's multi-scale features into a compact
representation Z (channels = cfg['z_channels']) at 1/8 resolution, plus the
multi-scale features themselves (passed to task heads / router).

This is deliberately simple (1x1 lateral convs + bilinear upsampling + add),
per the taskbook: no attention, no large FPN. The *representation* itself is
the research object (H2: one compact Z carries det + DA + lane information).
"""
import torch
import torch.nn as nn


class CompactRepresentation(nn.Module):
    def __init__(self, cfg, encoder_channels):
        """
        encoder_channels: list of encoder output channels [C2, C3, C4]
                          corresponding to [F2(1/8), F3(1/16), F4(1/32)].
        cfg: {"z_channels": int}
        """
        super().__init__()
        zc = cfg.get("z_channels", 64)
        c2, c3, c4 = encoder_channels

        self.lat4 = nn.Conv2d(c4, zc, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, zc, 1, bias=False)
        self.lat2 = nn.Conv2d(c2, zc, 1, bias=False)
        self.bn = nn.BatchNorm2d(zc)
        self.act = nn.ReLU(inplace=True)
        self.z_channels = zc

    def forward(self, feats):
        f2, f3, f4 = feats  # 1/8, 1/16, 1/32
        z = self.lat4(f4)
        z = nn.functional.interpolate(z, size=f3.shape[2:], mode="bilinear", align_corners=False)
        z = z + self.lat3(f3)
        z = nn.functional.interpolate(z, size=f2.shape[2:], mode="bilinear", align_corners=False)
        z = z + self.lat2(f2)
        z = self.act(self.bn(z))  # Z at 1/8
        return {"z": z, "multi_scale": feats}
