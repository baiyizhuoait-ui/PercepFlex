#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Phase 1-B figures (§22): Pareto, capacity curve, KD bar, budget dist."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "figures")
os.makedirs(OUT, exist_ok=True)


def load(p):
    pp = os.path.join(ROOT, p)
    return json.load(open(pp)) if os.path.exists(pp) else None


def main():
    # Figure 1/2: Pareto (FLOPs / latency vs tasks)
    rows = []
    for tag, path, flops in [
        ("static_tiny", "experiments/exp_static_vs_dynamic/static_tiny/metrics.json", 0.783),
        ("static_medium", "experiments/exp_static_vs_dynamic/static_medium/metrics.json", 0.924),
        ("static_large", "experiments/exp_static_vs_dynamic/static_large/metrics.json", 1.473),
        ("dynamic", "experiments/exp_static_vs_dynamic/dynamic/metrics.json", 0.90),
        ("static_eb", "experiments/expA_equal_budget/static_eb_s0_eval/metrics.json", 0.924),
    ]:
        m = load(path)
        if m:
            rows.append({"tag": tag, "flops": flops,
                         "lat": m.get("eval_fwd_ms", 0),
                         "mAP": m.get("mAP50", 0), "da": m.get("da_mIoU", 0),
                         "lane": m.get("lane_fg_iou", 0)})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in [(axes[0], "mAP", "Detection mAP50"),
                           (axes[1], "da", "DA mIoU"), (axes[2], "lane", "Lane fgIoU")]:
        for r in rows:
            ax.scatter(r["flops"], r[key], s=80, label=r["tag"])
        ax.set_xlabel("FLOPs (G)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 1: Accuracy-FLOPs Pareto")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_pareto_flops.png"), dpi=150)
    print("saved fig1")

    # Figure 3: capacity curve
    caps = []
    for tag, path in [
        ("0.23M", "experiments/expA_equal_budget/static_eb_s0_eval/metrics.json"),
        ("0.5M", "experiments/expD_compact_z/ours_0.5M_eval/metrics.json"),
        ("1.0M", "experiments/expD_compact_z/ours_1.0M_eval/metrics.json"),
        ("2.0M", "experiments/expD_compact_z/ours_2.0M_eval/metrics.json"),
    ]:
        m = load(path)
        if m:
            caps.append({"tag": tag, "mAP": m.get("mAP50", 0), "da": m.get("da_mIoU", 0),
                         "lane": m.get("lane_fg_iou", 0)})
    if caps:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(caps))
        for key, label in [("mAP", "mAP50"), ("da", "DA mIoU"), ("lane", "Lane fgIoU")]:
            ax.plot(x, [c[key] for c in caps], marker="o", label=label)
        ax.set_xticks(x, [c["tag"] for c in caps])
        ax.set_xlabel("Model capacity")
        ax.set_ylabel("Metric")
        ax.set_title("Figure 3: Compact Z capacity vs task performance")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig3_capacity.png"), dpi=150)
        print("saved fig3")


if __name__ == "__main__":
    main()
