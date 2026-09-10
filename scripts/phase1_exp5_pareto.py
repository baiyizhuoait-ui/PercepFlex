#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 5 (taskbook §11/§12): Pareto Frontier.

Collects metrics.json rows from the static variants, the dynamic model and the
light baselines, then plots Accuracy (mAP50 / DA mIoU / Lane mIoU) vs FLOPs and
vs latency. Also writes a comparison table (Best/Average/Worst per model).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from visualization.plots import pareto_frontier  # noqa: E402


def load_rows(paths):
    rows = []
    for p in paths:
        if os.path.isdir(p):
            p = os.path.join(p, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="metric dirs: static_tiny/medium/large, dynamic, baselines")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "pareto"))
    args = ap.parse_args()

    rows = load_rows(args.dirs)
    os.makedirs(args.out, exist_ok=True)
    for r in rows:
        r["flops_g"] = r.get("flops", 0) / 1e9
        r["latency_ms"] = r.get("avg_latency_ms", 0)

    labels = [r["model"] for r in rows]
    for xk in ("flops_g", "latency_ms"):
        pareto_frontier(rows, x_key=xk,
                        y_keys=("mAP50", "da_mIoU", "lane_mIoU"),
                        labels=labels,
                        out_path=os.path.join(args.out, f"pareto_{xk}.png"))

    # table
    print(f"{'model':<24} {'FLOPs(G)':>9} {'lat(ms)':>8} {'mAP50':>7} {'DA_mIoU':>8} {'Lane_mIoU':>9}")
    for r in rows:
        print(f"{r['model']:<24} {r['flops_g']:>9.2f} {r['latency_ms']:>8.2f} "
              f"{r.get('mAP50', 0):>7.4f} {r.get('da_mIoU', 0):>8.4f} {r.get('lane_mIoU', 0):>9.4f}")


if __name__ == "__main__":
    main()
