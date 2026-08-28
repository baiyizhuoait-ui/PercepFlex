"""Multi-task total loss with budget constraint (taskbook §9).

    L_total = L_task + lambda_budget * L_budget
    L_task  = L_det + lambda_da * L_da + lambda_lane * L_lane

L_budget constrains the expected computation of the dynamic model: it pushes the
router's expected widths toward a target envelope, so the model actually learns
to save compute rather than always choosing Large (failure mode A).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def seg_ce_loss(logits, mask, fg_weight=10.0):
    """logits: (B,2,H,W); mask: (B,1,H,W) binary float.
    fg_weight > 1 counters the extreme background dominance (lane fg ~1%)."""
    mask = mask.long().squeeze(1)
    w = torch.tensor([1.0, fg_weight], device=logits.device)
    return F.cross_entropy(logits, mask, weight=w, reduction="mean")


class MultiTaskLoss(nn.Module):
    def __init__(self, anchors, nc=1, lambda_da=1.0, lambda_lane=1.0,
                 lambda_budget=0.0, budget_target=None, budget_type="expected_width",
                 width_penalty=1.0, img_size=640):
        super().__init__()
        from losses.yolo_loss import YOLOLoss
        self.det_loss = YOLOLoss(anchors, nc=nc, img_size=img_size)
        self.lambda_da = lambda_da
        self.lambda_lane = lambda_lane
        self.lambda_budget = lambda_budget
        self.budget_target = budget_target  # e.g. 0.5 = average width envelope
        self.budget_type = budget_type
        self.width_penalty = width_penalty

    def forward(self, det_preds, da_logits, lane_logits,
                det_targets, da_mask, lane_mask, routing, img_size=640,
                ref=None, lambda_diff=0.0):
        """
        det_preds: list of per-scale raw (train mode) or None
        routing: router output dict with 'probs' (B,3,K)
        ref: (ref_det, ref_da, ref_lane) at a reference width — difficulty targets
        lambda_diff: weight of the difficulty-supervision loss (DifficultyRouter)
        """
        losses = {}
        if det_preds is not None:
            l_det, parts = self.det_loss(det_preds, det_targets, img_size)
            losses["det"] = l_det
        else:
            l_det = torch.zeros(1, device=da_logits.device).squeeze()
            losses["det"] = l_det
        l_da = seg_ce_loss(da_logits, da_mask)
        l_lane = seg_ce_loss(lane_logits, lane_mask)
        losses["da"], losses["lane"] = l_da, l_lane

        l_task = l_det + self.lambda_da * l_da + self.lambda_lane * l_lane
        losses["task"] = l_task

        # ---- difficulty supervision (DifficultyRouter) ----
        l_diff = torch.zeros(1, device=da_logits.device).squeeze()
        if ref is not None and routing is not None and "diff_pred" in routing and lambda_diff > 0:
            ref_det, ref_da, ref_lane = ref
            with torch.no_grad():
                l_det_r = self.det_loss(ref_det, det_targets, img_size)[0].detach() \
                    if ref_det is not None else l_det.detach()
                tgt = torch.stack([l_det_r,
                                   seg_ce_loss(ref_da, da_mask).detach(),
                                   seg_ce_loss(ref_lane, lane_mask).detach()])
                tgt = torch.log(tgt.clamp(min=1e-6))          # (3,)
            l_diff = F.mse_loss(routing["diff_pred"], tgt.unsqueeze(0).expand_as(routing["diff_pred"]))
        losses["diff"] = l_diff

        # ---- budget loss ----
        l_budget = torch.zeros(1, device=da_logits.device).squeeze()
        if routing is not None and self.lambda_budget > 0:
            probs = routing["probs"]                       # (B, 3, K)
            width_vec = torch.linspace(0.25, 1.0, probs.shape[-1], device=probs.device)
            exp_width = (probs * width_vec).sum(-1)        # (B, 3)
            if self.budget_type == "expected_width":
                if self.budget_target is not None:
                    l_budget = (exp_width.mean() - self.budget_target).abs() * self.width_penalty
                else:
                    l_budget = -((probs + 1e-8).log() * probs).sum(-1).mean()
        losses["budget"] = l_budget

        total = l_task + self.lambda_budget * l_budget + lambda_diff * l_diff
        losses["total"] = total
        return total, losses
