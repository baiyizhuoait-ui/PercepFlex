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
            self.det_p2 = det_cfg.get("p2", "none")
            self.det_head = DetFromZ(zc, nc=det_cfg.get("nc", 1),
                                     proj=det_cfg.get("z_proj", True),
                                     det_ch=det_cfg.get("det_ch", 32),
                                     anchors=det_cfg.get("anchors"),
                                     rec=det_cfg.get("rec", "1x1"),
                                     rec_blocks=det_cfg.get("rec_blocks", 2),
                                     rec_rates=tuple(det_cfg.get(
                                         "rec_rates", [2, 4])),
                                     p2=self.det_p2,
                                     p2_ch=enc_cfg["stages"][0])
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
        self.lane_head = DynamicSegHead(_w_lane, seg_cfg.get("lane_hidden", _hidden))
        # Phase 5: optional higher-resolution lane branch. Off by default.
        self.lane_res = int(seg_cfg.get("lane_res", 8))
        self.lane_use_f1 = bool(seg_cfg.get("lane_use_f1", False))
        # P4B-EXP-06: optional higher-resolution DA branch (mirror of lane). Off by default.
        self.da_res = int(seg_cfg.get("da_res", 8))
        self.da_use_f1 = bool(seg_cfg.get("da_use_f1", False))
        # P4B-EXP-03: p2="f1" also needs the encoder's native 1/4 map.
        self.need_highres = (bool(self.lane_res != 8 and self.lane_use_f1)
                             or self.det_p2 == "f1"
                             or bool(self.da_res != 8 and self.da_use_f1))
        if self.lane_use_f1:
            self.lane_lat = nn.Conv2d(enc_cfg["stages"][0], _w_lane, 1,
                                      bias=False)
        if self.da_use_f1:
            self.da_lat = nn.Conv2d(enc_cfg["stages"][0], _w_da, 1,
                                    bias=False)

    def forward(self, x, return_z=False):
        enc_out = self.encoder(x, highres=self.need_highres)
        if self.need_highres:
            feats, extra = enc_out
        else:
            feats, extra = enc_out, None
        z = self.representation(feats)["z"]
        hw = x.shape[2:]
        if self.det_from_z:
            det = self.det_head(z, hw,
                                extra=extra if self.det_p2 == "f1" else None)
        else:
            det = self.det_head([feats[0], feats[1], feats[2]])
        if self.task_proj:
            da_in = self.da_proj(z)
            lane_in = self.lane_proj(z)
        else:
            da_in = z
            lane_in = z
        if self.da_res != 8:
            s = 8 // self.da_res
            da_in = nn.functional.interpolate(
                da_in, size=(da_in.shape[-2] * s, da_in.shape[-1] * s),
                mode="bilinear", align_corners=False)
            if self.da_use_f1:
                da_in = da_in + self.da_lat(extra["f1"])
        da = self.da_head(da_in, hw)
        if self.lane_res != 8:
            s = 8 // self.lane_res
            lane_in = nn.functional.interpolate(
                lane_in, size=(lane_in.shape[-2] * s, lane_in.shape[-1] * s),
                mode="bilinear", align_corners=False)
            if self.lane_use_f1:
                lane_in = lane_in + self.lane_lat(extra["f1"])
        lane = self.lane_head(lane_in, hw)
        if return_z:
            return det, da, lane, z
        return det, da, lane

    def set_width(self, w):
        """Set dynamic width for all dynamic layers (w=1.0 == static)."""
        from models.router.dynamic_conv import set_width_recursive
        set_width_recursive(self, w)
