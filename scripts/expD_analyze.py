#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment D analysis: Compact Representation capacity scaling.

Answers (taskbook §D3):
  D3-1: as params shrink, does detection or segmentation degrade first?
  D3-2: are Lane / DA / Detection differently sensitive to capacity?
  D3-3: is there a minimal Z that still supports decent 3-task performance?
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAPS = [
    ("ours_0.23M", "experiments/expA_equal_budget/static_eb_s0_eval"),  # baseline compact
    ("ours_0.5M", "experiments/expD_compact_z/ours_0.5M_eval"),
    ("ours_1.0M", "experiments/expD_compact_z/ours_1.0M_eval"),
    ("ours_2.0M", "experiments/expD_compact_z/ours_2.0M_eval"),
]


def load_metrics(tag, path):
    p = os.path.join(ROOT, path, "metrics.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return {
        "model": tag,
        "params": m.get("parameters", 0) / 1e6,
        "flops_g": m.get("flops", 0) / 1e9,
        "mAP50": m.get("mAP50", 0),
        "da_mIoU": m.get("da_mIoU", 0),
        "lane_fg_iou": m.get("lane_fg_iou", 0),
    }


def main():
    rows = []
    for tag, path in CAPS:
        r = load_metrics(tag, path)
        if r:
            rows.append(r)
    if not rows:
        print("no D metrics yet (training in progress)")
        return
    print(f"{'model':<14} {'Params(M)':>9} {'FLOPs(G)':>9} {'mAP50':>7} {'DA_mIoU':>8} {'Lane_fgIoU':>10}")
    for r in rows:
        print(f"{r['model']:<14} {r['params']:>9.3f} {r['flops_g']:>9.2f} {r['mAP50']:>7.4f} "
              f"{r['da_mIoU']:>8.4f} {r['lane_fg_iou']:>10.4f}")
    if len(rows) >= 3:
        base = rows[0]
        print("\nrelative to smallest (normalized by largest params):")
        largest = rows[-1]
        for r in rows[:-1]:
            scale = largest["params"] / max(r["params"], 1e-6)
            print(f"  {r['model']}: 1/{scale:.1f}x params -> mAP {r['mAP50']/max(largest['mAP50'],1e-6):.2f}x, "
                  f"DA {r['da_mIoU']/max(largest['da_mIoU'],1e-6):.2f}x, "
                  f"Lane {r['lane_fg_iou']/max(largest['lane_fg_iou'],1e-6):.2f}x")
        # D3-1: which task degrades fastest when params shrink
        print("\nD3-1 (task degradation order as params shrink):")
        for task in ("mAP50", "da_mIoU", "lane_fg_iou"):
            ratios = [r[task] / max(rows[-1][task], 1e-6) for r in rows[:-1]]
            print(f"  {task}: retained fraction at each capacity = "
                  f"{['%.2f' % v for v in ratios]}")


if __name__ == "__main__":
    main()
