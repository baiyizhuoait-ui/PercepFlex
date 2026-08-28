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
from models.router.difficulty_router import DifficultyRouter
from models.heads.dynamic_seg_head import DynamicSegHead
from models.heads.dynamic_det_head import DynamicDetHead


def _set_head_width(head, w):
    from models.router.dynamic_conv import set_width_recursive
    set_width_recursive(head, w)


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
        router_cfg = model_cfg.get("router", {})
        if router_cfg.get("type", "task") == "difficulty":
            feat_dims = sum(2 * c for c in enc_cfg["stages"][1:])  # per-scale mean+std
            self.router = DifficultyRouter(
                zc, feat_dims, budgets=self.budgets,
                hidden=router_cfg.get("hidden", 64),
                shared=router_cfg.get("shared", False))
        else:
            self.router = TaskResourceRouter(
                zc, budgets=self.budgets,
                hidden=router_cfg.get("hidden", 64),
                shared=router_cfg.get("shared", False))
        self.router_type = router_cfg.get("type", "task")
        self.det_head = DynamicDetHead(
            enc_cfg["stages"][1:], nc=det_cfg.get("nc", 1),
            anchors=det_cfg.get("anchors"))
        seg_cfg = model_cfg.get("segmentation", {"hidden": 32})
        self.da_head = DynamicSegHead(zc, seg_cfg.get("hidden", 32))
        self.lane_head = DynamicSegHead(zc, seg_cfg.get("hidden", 32))
        # precompute dynamic-module lists for fast width switching
        self._head_dyn = {
            id(self.det_head): list(self.det_head.modules()),
            id(self.da_head): list(self.da_head.modules()),
            id(self.lane_head): list(self.lane_head.modules()),
        }
        self._last_head_w = {id(self.det_head): None, id(self.da_head): None, id(self.lane_head): None}

    def _set_width_fast(self, head_key, w):
        kid = id(head_key)
        if self._last_head_w[kid] == w:
            return
        self._last_head_w[kid] = w
        for m in self._head_dyn[kid]:
            if hasattr(m, "set_width"):
                m.set_width(w)
            elif hasattr(m, "width"):
                m.width = w

    def _extra_feats(self, feats, z):
        """Multi-scale mean/std statistics (difficulty signal for the router)."""
        parts = []
        for t in feats:
            parts.append(t.mean(dim=(2, 3)))
            parts.append(t.std(dim=(2, 3)))
        return torch.cat(parts, 1)

    def forward(self, x, budget_prior=None, hard=None, return_routing=False,
                force_widths=None, soft=False, ref_width=0.5, return_ref=False):
        """force_widths: (B,3) or (3,) tensor of widths to use instead of router
        output (used for static-width evaluation of the same model).

        soft=True (training): run each task head at ALL budgets and mix the
        outputs by the router's soft probabilities. This gives the router
        smooth gradients from every width (fixes the hard-STE blindness: the
        router sees how each width contributes to the loss, not just the
        sampled one). Eval always uses the discrete argmax path.

        return_ref=True (training, difficulty router): also run the heads at a
        reference width and return (ref_det, ref_da, ref_lane) so the loss can
        supervise the router's difficulty predictions.
        """
        feats = self.encoder(x)
        z = self.representation(feats)["z"]
        extra = self._extra_feats(feats, z) if self.router_type == "difficulty" else None
        routing = self.router(z, extra_feats=extra, budget_prior=budget_prior, hard=hard)
        if force_widths is not None:
            fw = force_widths.to(x.device)
            if fw.dim() == 1:
                fw = fw.unsqueeze(0).expand(x.shape[0], -1)
            budgets = self.router.budgets.to(x.device)
            assigns = torch.stack([
                (fw[:, t].unsqueeze(1) - budgets.unsqueeze(0)).abs().argmin(1)
                for t in range(3)], 1)
            routing = dict(routing)
            routing["widths"] = fw.to(x.device)
            routing["assignments"] = assigns

        ref = None
        if return_ref and self.training:
            self._set_width_fast(self.det_head, ref_width)
            self._set_width_fast(self.da_head, ref_width)
            self._set_width_fast(self.lane_head, ref_width)
            ref = (self.det_head([f for f in feats]),
                   self.da_head(z, x.shape[2:]), self.lane_head(z, x.shape[2:]))

        if soft and self.training:
            probs = routing["probs"]  # (B, 3, K)
            det = self._soft_mix(self.det_head, feats, probs[:, 0], is_det=True)
            da = self._soft_mix(self.da_head, z, probs[:, 1], target_hw=x.shape[2:])
            lane = self._soft_mix(self.lane_head, z, probs[:, 2], target_hw=x.shape[2:])
            if return_routing:
                return det, da, lane, routing, ref
            return det, da, lane, ref

        widths = routing["widths"]
        assignments = routing["assignments"]

        det = self._run_grouped(self.det_head, feats, widths[:, 0], assignments[:, 0],
                                n_out=1, is_det=True)
        da = self._run_grouped(self.da_head, z, widths[:, 1], assignments[:, 1],
                               n_out=2, target_hw=x.shape[2:])
        lane = self._run_grouped(self.lane_head, z, widths[:, 2], assignments[:, 2],
                                 n_out=2, target_hw=x.shape[2:])

        if return_ref and self.training:
            if return_routing:
                return det, da, lane, routing, ref
            return det, da, lane, ref
        if return_routing:
            return det, da, lane, routing
        return det, da, lane

    def _soft_mix(self, head, feats, p_task, target_hw=None, is_det=False):
        """Run head at every budget and mix outputs by per-sample soft probs.
        p_task: (B, K)."""
        outs = []
        for bi, budget in enumerate(self.budgets):
            self._set_width_fast(head, float(budget))
            if is_det:
                o = head([f for f in feats])
                # training det head returns list of per-scale raw tensors
                outs.append(o)
            else:
                outs.append(head(feats, target_hw))
        if is_det:
            nl = len(outs[0])
            return [sum(p_task[:, bi].view(-1, 1, 1, 1, 1) * outs[bi][s]
                        for bi in range(len(outs))) for s in range(nl)]
        return sum(p_task[:, bi].view(-1, 1, 1, 1) * outs[bi]
                   for bi in range(len(outs)))

    def _run_grouped(self, head, feats, widths, assignments, n_out, target_hw=None,
                     is_det=False):
        """Run `head` per unique assignment; outputs reordered to batch order.

        In training the det head returns a list of per-scale raw tensors;
        those are interleaved back into per-sample lists.
        """
        b = assignments.shape[0]
        if b == 1:
            self._set_width_fast(head, widths[0].item())
            if is_det:
                x = [t[0:1] for t in feats]
            else:
                x = feats[0:1]
            return head(x) if is_det else head(x, target_hw)

        out = [None] * b
        for a in assignments.unique():
            idx = (assignments == a).nonzero(as_tuple=False).flatten()
            self._set_width_fast(head, widths[idx[0]].item())
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
        for h in (self.det_head, self.da_head, self.lane_head):
            self._set_width_fast(h, w)

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
