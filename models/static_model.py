"""Static multi-task model (Phase 1 Step 5).

Input -> LightEncoder -> multi-scale [F2,F3,F4] -> CompactRepresentation Z
     -> DetHead (multi-scale) | SegHead-DA (Z) | SegHead-Lane (Z)

Uses the SAME dynamic-width heads as the adaptive model at width=1.0, so a
Stage-A checkpoint transfers directly into the adaptive model (Stage B init).
Inference output format matches YOLOP/TriLiteNet:
    det: (B, 25200, 5+nc) decoded xywh+obj+cls
    da / lane: (B, 2, H, W) logits
"""
import torch
import torch.nn as nn

from models.encoder.light_encoder import LightEncoder
from models.representation.compact_z import CompactRepresentation
from models.heads.dynamic_det_head import DynamicDetHead
from models.heads.dynamic_seg_head import DynamicSegHead


class StaticMultiTaskModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        enc_cfg = model_cfg["encoder"]
        self.encoder = LightEncoder(enc_cfg)
        self.representation = CompactRepresentation(
            model_cfg.get("representation", {"z_channels": 64}),
            enc_cfg["stages"][1:])
        zc = self.representation.z_channels
        det_cfg = model_cfg.get("detection", {})
        self.det_head = DynamicDetHead(
            enc_cfg["stages"][1:],
            nc=det_cfg.get("nc", 1),
            anchors=det_cfg.get("anchors"))
        seg_cfg = model_cfg.get("segmentation", {"hidden": 32})
        self.da_head = DynamicSegHead(zc, seg_cfg.get("hidden", 32))
        self.lane_head = DynamicSegHead(zc, seg_cfg.get("hidden", 32))

    def forward(self, x):
        feats = self.encoder(x)
        z = self.representation(feats)["z"]
        det = self.det_head([feats[0], feats[1], feats[2]])
        hw = x.shape[2:]
        da = self.da_head(z, hw)
        lane = self.lane_head(z, hw)
        return det, da, lane

    def set_width(self, w):
        """Set dynamic width for all dynamic layers (w=1.0 == static)."""
        for m in self.modules():
            if hasattr(m, "width"):
                m.width = w
