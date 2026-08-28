"""Visualization utilities (Phase 1 Step 12).

- pareto_frontier: Accuracy vs FLOPs/latency scatter for static models and the
  dynamic model (per-frame operating points).
- allocation_plots: per-frame task budget bars + budget distribution histograms
  (taskbook §14) — prove the router behavior is semantic, not random.
- failure_stats: the five failure modes of taskbook §13.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pareto_frontier(rows, x_key="flops_g", y_keys=("mAP50", "da_mIoU", "lane_mIoU"),
                    labels=None, out_path=None, title="Accuracy vs Compute"):
    """rows: list of dicts with x and y metrics (static models + dynamic means)."""
    fig, axes = plt.subplots(1, len(y_keys), figsize=(5 * len(y_keys), 4.5))
    if len(y_keys) == 1:
        axes = [axes]
    for ax, yk in zip(axes, y_keys):
        for r, lab in zip(rows, labels or range(len(rows))):
            ax.scatter(r[x_key], r.get(yk, 0), s=60, label=str(lab))
        ax.set_xlabel(x_key)
        ax.set_ylabel(yk)
        ax.set_title(f"{title} ({yk})")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"saved -> {out_path}")
    return fig


def allocation_plots(csv_path, budgets=(0.25, 0.5, 1.0), out_dir=None,
                     max_frames=200):
    """Per-frame task budget bars + distribution histograms from
    allocation_statistics.csv (taskbook §14)."""
    import csv

    rows = list(csv.DictReader(open(csv_path)))
    frames = [r["frame"] for r in rows][:max_frames]
    det = [int(r["det"]) for r in rows][:max_frames]
    da = [int(r["da"]) for r in rows][:max_frames]
    lane = [int(r["lane"]) for r in rows][:max_frames]
    x = np.arange(len(frames))
    w = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    ax.bar(x - w, det, w, label="det", color="#d62728")
    ax.bar(x, da, w, label="da", color="#1f77b4")
    ax.bar(x + w, lane, w, label="lane", color="#2ca02c")
    ax.set_xlabel("frame")
    ax.set_ylabel("budget index")
    ax.set_yticks(range(len(budgets)), [f"{b:g}" for b in budgets])
    ax.set_title("Per-frame task allocation")
    ax.legend()
    ax.set_xticks(x[:: max(1, len(x) // 10)])

    ax = axes[1]
    for i, (data, name) in enumerate([(det, "det"), (da, "da"), (lane, "lane")]):
        counts = np.bincount(data, minlength=len(budgets)) / len(data)
        ax.bar(np.arange(len(budgets)) + i * 0.25, counts, 0.2, label=name)
    ax.set_xticks(range(len(budgets)), [f"{b:g}" for b in budgets])
    ax.set_xlabel("budget")
    ax.set_ylabel("fraction")
    ax.set_title("Budget distribution per task")
    ax.legend()
    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "allocation.png")
        fig.savefig(p, dpi=150)
        print(f"saved -> {p}")
    return fig


def failure_stats(alloc_rows, budgets=3):
    """Taskbook §13 failure-mode statistics from allocation data."""
    n = len(alloc_rows)
    if n == 0:
        return {}
    det = np.array([r["det"] for r in alloc_rows])
    da = np.array([r["da"] for r in alloc_rows])
    lane = np.array([r["lane"] for r in alloc_rows])
    all_tasks = np.stack([det, da, lane], 1)
    return {
        "n_frames": n,
        "frac_large_det": float((det == budgets - 1).mean()),
        "frac_large_da": float((da == budgets - 1).mean()),
        "frac_large_lane": float((lane == budgets - 1).mean()),
        "frac_tiny_det": float((det == 0).mean()),
        "frac_tiny_da": float((da == 0).mean()),
        "frac_tiny_lane": float((lane == 0).mean()),
        "frac_same_all_tasks": float((all_tasks[:, 0] == all_tasks[:, 1]).mean()
                                     * (all_tasks[:, 1] == all_tasks[:, 2]).mean()),
        "task_heterogeneity": float((all_tasks.max(1) != all_tasks.min(1)).mean()),
    }
