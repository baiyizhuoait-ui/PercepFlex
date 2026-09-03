"""Multi-Teacher Cross-Architecture KD (Phase 2 Step 3, minimal v1).

Per taskbook §7-9 — deliberately simple, no attention / teacher-fusion network:

    Teacher_i feature (1/8)
         |  P_i (1x1 projection)     # teacher-specific, one per teacher
         v
         z_i  (common latent space, = student z_ch)
         |  fuse (Method 1 avg / Method 2 fixed weights / Method 3 conf proxy)
         v
    z_common  --> align  student Compact Z (MSE)

Plus per-task output KD: student seg logits vs. (fused) teacher seg probabilities.

Design notes / boundary (taskbook §23, §10):
- NOT a learned attention fusion; teacher fusion here is a plain weighted combine.
- Each teacher keeps its OWN P_i (their channel/statistics differ) -> "common latent".
- Student Z is shared and feeds the seg heads; detection improves indirectly via
  the better shared representation (same mechanism as the Phase-1 single-teacher KD).
- Fusion of FEATURES only. Seg output KD uses an equal or task-aware mix of teachers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.kd import KDProjection, TeacherAdapter


def fuse(features, weights=None):
    """features: list[Tensor (B,C,H,W)]; weights: optional per-teacher (len N).
    Returns (B,C,H,W) combined latent + dict of per-teacher contributions."""
    if weights is None:
        z = torch.stack(features, 0).mean(0)
    else:
        w = torch.tensor(weights, dtype=features[0].dtype, device=features[0].device)
        w = w / w.sum()  # normalize
        z = sum(wi * fi for wi, fi in zip(w, features))
    return z


class MultiTeacherKD(nn.Module):
    """N frozen teachers -> per-teacher projection -> fused common latent for student Z.

    teachers: list of (name, model, norm) e.g.
        [("YOLOP", yolop, "imagenet"), ("TwinLiteNetPlus", tlp, "unit"),
         ("TriLiteNet", tri, "imagenet")]
    """

    def __init__(self, teachers, z_ch=64, lambda_feat=0.1, lambda_out=1.0,
                 method="avg", weights=None):
        super().__init__()
        self.z_ch = z_ch
        self.lambda_feat = lambda_feat
        self.lambda_out = lambda_out
        self.method = method
        self.weights = weights  # fixed per-teacher weights (Method 2); None => equal
        self.names = [t[0] for t in teachers]
        # frozen per-teacher feature extractors
        self.adapters = nn.ModuleList(
            TeacherAdapter(name, model, norm) for name, model, norm in teachers)
        # one projection per teacher -> common latent (lazy: needs feat channels)
        self.projs = nn.ModuleList()

    def _ensure_projs(self, feat_chs, device):
        if len(self.projs) == 0:
            for c in feat_chs:
                self.projs.append(KDProjection(c, self.z_ch).to(device))

    def forward(self, x, student_z, student_da, student_lane):
        feats, da_t_all, lane_t_all = [], [], []
        with torch.no_grad():
            for ad in self.adapters:
                f, da, lane = ad(x)
                feats.append(f)
                da_t_all.append(da)
                lane_t_all.append(lane)
        feat_chs = [f.shape[1] for f in feats]
        self._ensure_projs(feat_chs, x.device)
        # teacher-specific projection -> common latent space, then fuse
        zs = []
        for P, f in zip(self.projs, feats):
            zi = P(f)
            if zi.shape[2:] != student_z.shape[2:]:
                zi = F.interpolate(zi, size=student_z.shape[2:],
                                   mode="bilinear", align_corners=False)
            zs.append(zi)
        if self.method == "avg":
            z_common = fuse(zs)
        else:  # "weighted" (Method 2: fixed task-aware presets come via `weights`)
            z_common = fuse(zs, self.weights)

        losses = {}
        if student_z is not None:
            losses["feat"] = F.mse_loss(z_common, student_z) * self.lambda_feat
        # output KD: average teacher seg probabilities per task -> student
        if student_da is not None and da_t_all:
            t = torch.stack(da_t_all, 0)
            p_t = torch.sigmoid(t.mean(0))          # fused teacher probability
            losses["out_da"] = F.mse_loss(torch.sigmoid(student_da), p_t) * self.lambda_out
        if student_lane is not None and lane_t_all:
            t = torch.stack(lane_t_all, 0)
            p_t = torch.sigmoid(t.mean(0))
            losses["out_lane"] = F.mse_loss(torch.sigmoid(student_lane), p_t) * self.lambda_out
        return losses
