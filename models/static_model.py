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
        _tp = model_cfg.get("task_proj") or {}
        _tp_on = bool(_tp.get("enabled", False))
        if _tp_on and "det" in _tp:
            det_cfg = dict(det_cfg)
            det_cfg["det_ch"] = int(_tp["det"])
        if det_cfg.get("from_z", False):
            # R1/R2: detection reads Compact Z (optionally via a 1x1 projection).
            from models.representation.det_from_z import DetFromZ
            self.det_head = DetFromZ(zc, nc=det_cfg.get("nc", 1),
                                     proj=det_cfg.get("z_proj", True),
                                     det_ch=det_cfg.get("det_ch", 32),
                                     anchors=det_cfg.get("anchors"),
                                     rec=det_cfg.get("rec", "1x1"),
                                     rec_blocks=det_cfg.get("rec_blocks", 2),
                                     rec_rates=tuple(det_cfg.get(
                                         "rec_rates", [2, 4])))
            self.det_from_z = True
        else:
            self.det_head = DynamicDetHead(
                enc_cfg["stages"][1:],
                nc=det_cfg.get("nc", 1),
                anchors=det_cfg.get("anchors"))
            self.det_from_z = False
        seg_cfg = model_cfg.get("segmentation", {"hidden": 32})
        _hidden = seg_cfg.get("hidden", 32)
        # Phase 4A Level 2 probe A: per-task projection off the shared Z.
        # Default off -> existing R0/R1/R2 models build exactly as before.
        self.task_proj = _tp_on
        if self.task_proj:
            from models.representation.task_proj import TaskProj
            self.da_proj = TaskProj(zc, int(_tp.get("da", zc)))
            self.lane_proj = TaskProj(zc, int(_tp.get("lane", zc)))
            _w_da = self.da_proj.out_ch
            _w_lane = self.lane_proj.out_ch
        else:
            _w_da = zc
            _w_lane = zc
        self.da_head = DynamicSegHead(_w_da, _hidden)
        self.lane_head = DynamicSegHead(_w_lane, _hidden)

    def forward(self, x, return_z=False):
        feats = self.encoder(x)
        z = self.representation(feats)["z"]
        hw = x.shape[2:]
        if self.det_from_z:
            det = self.det_head(z, hw)
        else:
            det = self.det_head([feats[0], feats[1], feats[2]])
        if self.task_proj:
            da = self.da_head(self.da_proj(z), hw)
            lane = self.lane_head(self.lane_proj(z), hw)
        else:
            da = self.da_head(z, hw)
            lane = self.lane_head(z, hw)
        if return_z:
            return det, da, lane, z
        return det, da, lane

    def set_width(self, w):
        """Set dynamic width for all dynamic layers (w=1.0 == static)."""
        from models.router.dynamic_conv import set_width_recursive
        set_width_recursive(self, w)
