"""Adaptive multi-task model (Phase 1 Steps 6-8).

Assembles the full Task x Resource x Representation stack:

    Input
      -> LightEncoder -> multi-scale [F2,F3,F4]
      -> CompactRepresentation Z
      -> TaskResourceRouter(Z, budget_prior) -> (alpha_det, alpha_da, alpha_lane)
      -> dynamic task heads run at their allocated width (nested-prefix slicing)

Supports per-image task-wise heterogeneous allocation: within one batch,
samples are grouped by their (det, da, lane) assignment and each group runs
its heads at the assigned widths. In inference (hard, bs can be 1 or grouped).

Training-time: Gumbel-softmax (hard) for differentiable discrete selection.
"""
import torch
import torch.nn as nn

from models.encoder.light_encoder import LightEncoder
from models.representation.compact_z import CompactRepresentation
from models.router.task_resource_router import TaskResourceRouter
from models.heads.dynamic_seg_head import DynamicSegHead
from models.heads.dynamic_det_head import DynamicDetHead


def _set_head_width(head, w):
    for m in head.modules():
        if hasattr(m, "width"):
            m.width = w


class AdaptiveMultiTaskModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        enc_cfg = model_cfg["encoder"]
        self.encoder = LightEncoder(enc_cfg)
        self.representation = CompactRepresentation(
            model_cfg.get("representation", {"z_channels": 64}),
            enc_cfg["stages"][1:])
        zc = self.representation.z_channels
        det_cfg = model_cfg.get("detection", {})
        self.budgets = tuple(model_cfg.get("budgets", (0.25, 0.5, 1.0)))
        self.router = TaskResourceRouter(
            zc, budgets=self.budgets,
            hidden=model_cfg.get("router", {}).get("hidden", 64),
            shared=model_cfg.get("router", {}).get("shared", False))
        self.det_head = DynamicDetHead(
            enc_cfg["stages"][1:], nc=det_cfg.get("nc", 1),
            anchors=det_cfg.get("anchors"))
        seg_cfg = model_cfg.get("segmentation", {"hidden": 32})
        self.da_head = DynamicSegHead(zc, seg_cfg.get("hidden", 32))
        self.lane_head = DynamicSegHead(zc, seg_cfg.get("hidden", 32))

    def forward(self, x, budget_prior=None, hard=None, return_routing=False,
                force_widths=None):
        """force_widths: (B,3) or (3,) tensor of widths to use instead of router
        output (used for static-width evaluation of the same model)."""
        feats = self.encoder(x)
        z = self.representation(feats)["z"]
        routing = self.router(z, budget_prior=budget_prior, hard=hard)
        if force_widths is not None:
            fw = force_widths.to(x.device)
            if fw.dim() == 1:
                fw = fw.unsqueeze(0).expand(x.shape[0], -1)
            budgets = self.router.budgets.to(x.device)
            # map forced widths to nearest budget indices so batching/grouping works
            assigns = torch.stack([
                (fw[:, t].unsqueeze(1) - budgets.unsqueeze(0)).abs().argmin(1)
                for t in range(3)], 1)
            routing = dict(routing)
            routing["widths"] = fw.to(x.device)
            routing["assignments"] = assigns
        widths = routing["widths"]           # (B, 3): det, da, lane
        assignments = routing["assignments"]  # (B, 3)

        det = self._run_grouped(self.det_head, feats, widths[:, 0], assignments[:, 0],
                                n_out=1, is_det=True)
        da = self._run_grouped(self.da_head, z, widths[:, 1], assignments[:, 1],
                               n_out=2, target_hw=x.shape[2:])
        lane = self._run_grouped(self.lane_head, z, widths[:, 2], assignments[:, 2],
                                 n_out=2, target_hw=x.shape[2:])

        if return_routing:
            return det, da, lane, routing
        return det, da, lane

    def _run_grouped(self, head, feats, widths, assignments, n_out, target_hw=None,
                     is_det=False):
        """Run `head` per unique assignment; outputs reordered to batch order.

        In training the det head returns a list of per-scale raw tensors;
        those are interleaved back into per-sample lists.
        """
        b = assignments.shape[0]
        if b == 1:
            _set_head_width(head, widths[0].item())
            if is_det:
                x = [t[0:1] for t in feats]
            else:
                x = feats[0:1]
            return head(x) if is_det else head(x, target_hw)

        out = [None] * b
        for a in assignments.unique():
            idx = (assignments == a).nonzero(as_tuple=False).flatten()
            _set_head_width(head, widths[idx[0]].item())
            if is_det:
                sub = [torch.stack([t[i] for i in idx.tolist()]) for t in feats]
                o = head(sub)
                if isinstance(o, (list, tuple)):  # training: per-scale raw
                    for j, i in enumerate(idx.tolist()):
                        out[i] = [o_l[j:j + 1] for o_l in o]
                else:                              # eval: (B, 25200, no)
                    for j, i in enumerate(idx.tolist()):
                        out[i] = o[j]
            else:
                sub = torch.stack([feats[i] for i in idx.tolist()])
                o = head(sub, target_hw)
                for j, i in enumerate(idx.tolist()):
                    out[i] = o[j]
        if isinstance(out[0], list):  # training det: concat per scale
            return [torch.cat([out[i][s] for i in range(b)], 0)
                    for s in range(len(out[0]))]
        return torch.stack(out)

    def set_width(self, w):
        for head in (self.det_head, self.da_head, self.lane_head):
            _set_head_width(head, w)

    def set_mode(self, mode, strength=1.5):
        """Runtime mode: HIGH / MEDIUM / LOW -> budget prior bias."""
        self.mode = mode
        self.mode_strength = strength

    def forward_mode(self, x, mode=None):
        prior = None
        if mode is not None:
            prior = TaskResourceRouter.budget_prior_from_mode(
                mode, len(self.budgets), self.mode_strength, x.device)
        return self.forward(x, budget_prior=prior, hard=True)
