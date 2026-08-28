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
                det_targets, da_mask, lane_mask, routing, img_size=640):
        """
        det_preds: list of per-scale raw (train mode) or None
        routing: router output dict with 'probs' (B,3,K)
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

        # ---- budget loss ----
        l_budget = torch.zeros(1, device=da_logits.device).squeeze()
        if routing is not None and self.lambda_budget > 0:
            probs = routing["probs"]                       # (B, 3, K)
            budgets = routing["one_hot"].new_tensor([0.25, 0.5, 1.0] if False else None) if False else None
            # expected width per task from soft probs
            width_vec = torch.linspace(0.25, 1.0, probs.shape[-1], device=probs.device)
            exp_width = (probs * width_vec).sum(-1)        # (B, 3)
            if self.budget_type == "expected_width":
                if self.budget_target is not None:
                    # push mean expected width toward target (compute envelope)
                    l_budget = (exp_width.mean() - self.budget_target).abs() * self.width_penalty
                else:
                    # entropy regularization: encourage decisive routing
                    l_budget = -((probs + 1e-8).log() * probs).sum(-1).mean()
        losses["budget"] = l_budget

        total = l_task + self.lambda_budget * l_budget
        losses["total"] = total
        return total, losses
