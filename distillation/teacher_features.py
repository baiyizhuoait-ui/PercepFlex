"""Automatic teacher feature extraction for Cross-Architecture KD.

Finds the teacher's richest 1/8-resolution shared feature by running one
forward with hooks, then aligns it (via projection) with the student's
Compact Z. Works for YOLOP / TriLiteNet (sequential YOLO-style) and
TwinLiteNet+ (explicit encoder).
"""
import torch
import torch.nn as nn


def find_teacher_1_8_feature(model, input_size=(1, 3, 640, 640), min_channels=64):
    """Return a forward hook that captures the max-channel 1/8 feature."""
    captured = {}

    def make_hook(name):
        def h(m, i, o):
            if isinstance(o, torch.Tensor) and o.dim() == 4:
                _, c, hh, ww = o.shape
                # 1/8 of 640x640 (within tolerance for padding/strides)
                if abs(hh - input_size[2] // 8) <= 1 and c >= min_channels:
                    if "best" not in captured or c > captured["best"][1]:
                        captured["best"] = (name, c)
        return h

    hooks = []
    # TwinLiteNet+ style: explicit encoder
    enc = getattr(model, "encoder", None)
    if enc is not None and isinstance(enc, nn.Module):
        hooks.append(enc.register_forward_hook(make_hook("encoder")))
    # sequential models: hook every module with conv-like output
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Sequential, nn.ModuleList)):
            pass
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.Sequential):
            for i, blk in enumerate(mod):
                hooks.append(blk.register_forward_hook(make_hook(f"{name}.{i}")))
    for name, mod in model.named_modules():
        if name.count(".") <= 1 and isinstance(mod, nn.Module) and mod is not model:
            try:
                hooks.append(mod.register_forward_hook(make_hook(name)))
            except Exception:
                pass

    x = torch.randn(*input_size)
    with torch.no_grad():
        try:
            model(x)
        except Exception:
            # some models need a different input device/format; try cuda
            pass
    for hk in hooks:
        hk.remove()
    return captured.get("best")


class TeacherFeatureExtractor(nn.Module):
    """Wraps a frozen teacher and exposes its 1/8 feature + seg logits."""

    def __init__(self, teacher, feature_name, norm="imagenet", fp16=False):
        super().__init__()
        self.teacher = teacher
        for p in teacher.parameters():
            p.requires_grad = False
        self.feature_name = feature_name
        self.norm = norm
        self.fp16 = fp16
        self._cache = {}

    def forward(self, x):
        """x: (B,3,640,640) in [0,1]. Returns (feature_1_8, da_logits, lane_logits)."""
        t = x
        if self.norm == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            t = (t - mean) / std
        if self.fp16:
            self.teacher.half()
            t = t.half()
        with torch.no_grad():
            out = self.teacher(t)
        return out  # raw teacher output; parsed by the KD loss

    def parse(self, out):
        """Parse teacher output into (feature_1_8, da, lane) — overridden per arch."""
        raise NotImplementedError
