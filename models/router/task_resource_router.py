"""Task-Resource Router (Module B — the core innovation of Phase 1).

Learns, per frame, *which task is hard* (task-wise heterogeneous allocation),
not merely "is this frame hard". Input is the compact representation Z;
output is a discrete per-task budget assignment (alpha_det, alpha_da, alpha_lane).

    Z -> GlobalAvgPool -> small MLP -> per-task allocation logits
        -> (optional) budget prior bias (LOW/MEDIUM/HIGH constraint)
        -> discrete selection (Gumbel-softmax in training, argmax in eval)
        -> per-task widths in {budgets}

budgets: nested-prefix channel ratios, e.g. [0.25, 0.5, 1.0] (Tiny/Medium/Large).
The budget prior lets an operator constrain the overall compute envelope:
    LOW    -> bias towards tiny budgets
    MEDIUM -> bias towards medium
    HIGH   -> bias towards large
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskResourceRouter(nn.Module):
    def __init__(self, z_channels, budgets=(0.25, 0.5, 1.0), hidden=64,
                 temperature=1.0):
        super().__init__()
        self.budgets = torch.tensor(list(budgets), dtype=torch.float32)
        self.n_tasks = 3  # det, da, lane
        self.n_budgets = len(budgets)
        self.temperature = temperature
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_tasks * self.n_budgets),
        )
        # optional bias term per (task, budget): initialized small
        self.prior_bias = nn.Parameter(torch.zeros(self.n_tasks, self.n_budgets))

    def forward(self, z, budget_prior=None, hard=None):
        """
        z: (B, zc, h, w)
        budget_prior: (B, n_budgets) or (n_budgets,) logit bias for the overall
                      compute envelope (None = free allocation).
        Returns dict with logits, probs, widths (B,3), assignments (B,3).
        """
        g = self.pool(z).flatten(1)                       # (B, zc)
        logits = self.mlp(g).view(-1, self.n_tasks, self.n_budgets)  # (B,3,K)
        logits = logits + self.prior_bias[None]
        if budget_prior is not None:
            logits = logits + budget_prior.unsqueeze(1)   # (B,1,K) broadcast
        probs = torch.softmax(logits, dim=-1)

        if hard is None:
            hard = not self.training
        if self.training or hard:
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
            widths = one_hot @ self.budgets.to(logits.device)     # (B,3)
            assignments = one_hot.argmax(-1)
        else:
            assignments = logits.argmax(-1)
            widths = self.budgets.to(logits.device)[assignments]
            one_hot = F.one_hot(assignments, self.n_budgets).float()
        return {
            "logits": logits,
            "probs": probs,
            "one_hot": one_hot,
            "widths": widths,          # (B,3): alpha_det, alpha_da, alpha_lane
            "assignments": assignments,  # (B,3): budget indices
        }

    @staticmethod
    def budget_prior_from_mode(mode, n_budgets=3, strength=1.5, device="cpu"):
        """Encode LOW/MEDIUM/HIGH into a logit bias."""
        m = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[mode]
        bias = torch.zeros(n_budgets, device=device)
        bias[m] = strength
        return bias
