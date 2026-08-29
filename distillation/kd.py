"""Cross-Architecture Knowledge Distillation (Phase 1-B, minimal v1).

Teacher feature -> 1x1 projection -> student Compact Z (L2 align at 1/8)
Teacher DA/Lane logits -> student logits (KL, temperature T)
No attention / multi-layer alignment / teacher ensembles (taskbook §10).

Teachers:
  YOLOP        : hook model[16]        -> (1,256,80,80); da/lane = out[1:3]
  TriLiteNet   : hook model[0] (tuple) -> [0] (1,256,80,80); da/lane = out[1:3]
  TwinLiteNet+ : hook encoder (tuple)  -> [0] (1,C,80,80);  da/lane = out
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KDProjection(nn.Module):
    """1x1 conv + BN projecting a teacher feature to the student Z space."""

    def __init__(self, in_ch, z_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, z_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(z_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


def _hook_teacher(name, teacher):
    """Register a forward hook returning (feature_1_8, da, lane) extraction."""
    caps = {}

    def capture(key):
        def h(m, i, o):
            caps[key] = o
        return h

    if name == "YOLOP":
        teacher.model[16].register_forward_hook(capture("feat"))
        feats = caps
        return feats
    return caps


class TeacherAdapter(nn.Module):
    """Frozen teacher with a hook; exposes feature_1_8 + seg logits."""

    def __init__(self, name, teacher, norm="imagenet"):
        super().__init__()
        self.name = name
        self.teacher = teacher
        self.norm = norm
        for p in teacher.parameters():
            p.requires_grad = False
        self._feat = {}

        if name == "YOLOP":
            teacher.model[16].register_forward_hook(self._hook("feat"))
        elif name == "TriLiteNet":
            teacher.model[0].register_forward_hook(self._hook("enc"))
        elif name == "TwinLiteNetPlus":
            teacher.encoder.register_forward_hook(self._hook("enc"))

    def _hook(self, key):
        def h(m, i, o):
            self._feat[key] = o
        return h

    def forward(self, x):
        """x: (B,3,640,640) in [0,1]. Returns (feature_1_8, da_logits, lane_logits)."""
        t = x
        if self.norm == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            t = (t - mean) / std
        with torch.no_grad():
            out = self.teacher(t)
        if self.name == "YOLOP":
            feat = self._feat.get("feat")
            da, lane = out[1], out[2]
        elif self.name == "TriLiteNet":
            enc = self._feat.get("enc")
            feat = enc[0] if isinstance(enc, (tuple, list)) else enc
            da, lane = out[1], out[2]
        elif self.name == "TwinLiteNetPlus":
            enc = self._feat.get("enc")
            feat = enc[0] if isinstance(enc, (tuple, list)) else enc
            da, lane = out[0], out[1]
        self._feat.clear()
        return feat, da, lane


def kl_div_logits(student_logits, teacher_logits, T=4.0):
    """Robust output KD: sigmoid-MSE on probability maps.

    Teachers differ: YOLOP emits sigmoid probabilities, TriLiteNet/TLP emit
    logits. Normalizing both to probabilities and using MSE is scale-stable
    across teachers and avoids the KL blow-up on confident teachers.
    """
    if student_logits.shape != teacher_logits.shape:
        teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[2:],
                                       mode="bilinear", align_corners=False)
    p_stu = torch.sigmoid(student_logits)
    p_tea = torch.sigmoid(teacher_logits)
    return F.mse_loss(p_stu, p_tea)


class CrossArchKD(nn.Module):
    def __init__(self, teacher_name, teacher, z_ch, lambda_feat=0.1,
                 lambda_out=1.0, T=4.0, norm="imagenet"):
        super().__init__()
        self.adapter = TeacherAdapter(teacher_name, teacher, norm)
        self.lambda_feat = lambda_feat
        self.lambda_out = lambda_out
        self.T = T
        # projection created lazily (needs teacher feature channels)
        self.proj = None

    def _ensure_proj(self, feat_ch, device):
        if self.proj is None:
            self.proj = KDProjection(feat_ch, 64).to(device)

    def forward(self, x, student_z, student_da, student_lane):
        feat, da_t, lane_t = self.adapter(x)
        losses = {}
        if feat is not None and student_z is not None:
            self._ensure_proj(feat.shape[1], x.device)
            z_t = self.proj(feat)
            if z_t.shape[2:] != student_z.shape[2:]:
                z_t = F.interpolate(z_t, size=student_z.shape[2:],
                                    mode="bilinear", align_corners=False)
            losses["feat"] = F.mse_loss(z_t, student_z) * self.lambda_feat
        if student_da is not None and da_t is not None:
            losses["out_da"] = kl_div_logits(student_da, da_t, self.T) * self.lambda_out
        if student_lane is not None and lane_t is not None:
            losses["out_lane"] = kl_div_logits(student_lane, lane_t, self.T) * self.lambda_out
        return losses
