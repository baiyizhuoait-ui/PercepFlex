"""Difficulty-Supervised Task-Resource Router (core fix v2).

Learns per-frame, per-task difficulty from RICH shared-feature statistics
(multi-scale mean/std — the probe showed AUC 0.69 vs 0.5 for pooled-Z only),
and receives DIRECT supervision: an auxiliary difficulty head predicts the
per-task losses (computed at a reference width during training), so the router
learns "which task is hard for this frame" explicitly instead of relying on the
noisy task-loss gradient through discrete choices.

    Z + multi-scale stats -> MLP -> [routing logits (3,K)]  +  [difficulty pred (3)]
    L_router = L_task(budget-aware) + lambda_aux * MSE(diff_pred, log L_task_ref)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyRouter(nn.Module):
    def __init__(self, z_channels, feat_dims, budgets=(0.25, 0.5, 1.0),
                 hidden=64, temperature=1.0, shared=False):
        """
        feat_dims: extra per-scale statistic dims concatenated to pooled Z.
        """
        super().__init__()
        self.budgets = torch.tensor(list(budgets), dtype=torch.float32)
        self.n_tasks = 1 if shared else 3
        self.n_budgets = len(budgets)
        self.temperature = temperature
        self.shared = shared
        in_dim = z_channels + feat_dims
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.route = nn.Linear(hidden, self.n_tasks * self.n_budgets)
        self.diff_head = nn.Linear(hidden, self.n_tasks)   # predicted log task loss
        self.prior_bias = nn.Parameter(torch.zeros(self.n_tasks, self.n_budgets))

    def forward(self, z, extra_feats=None, budget_prior=None, hard=None):
        g = self.pool(z).flatten(1)
        if extra_feats is not None:
            g = torch.cat([g, extra_feats], 1)
        h = self.mlp(g)
        logits = self.route(h).view(-1, self.n_tasks, self.n_budgets)
        logits = logits + self.prior_bias[None]
        diff_pred = self.diff_head(h)                       # (B, n_tasks)
        if budget_prior is not None:
            logits = logits + budget_prior.unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)

        if hard is None:
            hard = not self.training
        if self.training or hard:
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
            widths = one_hot @ self.budgets.to(logits.device)
            assignments = one_hot.argmax(-1)
        else:
            assignments = logits.argmax(-1)
            widths = self.budgets.to(logits.device)[assignments]
            one_hot = F.one_hot(assignments, self.n_budgets).float()

        if self.shared:
            widths = widths.repeat(1, 3)
            assignments = assignments.repeat(1, 3)
            one_hot = one_hot.repeat(1, 3, 1)
            diff_pred = diff_pred.repeat(1, 3)

        return {
            "logits": logits, "probs": probs, "one_hot": one_hot,
            "widths": widths, "assignments": assignments,
            "diff_pred": diff_pred,
        }

    @staticmethod
    def budget_prior_from_mode(mode, n_budgets=3, strength=1.5, device="cpu"):
        m = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[mode]
        bias = torch.zeros(n_budgets, device=device)
        bias[m] = strength
        return bias
