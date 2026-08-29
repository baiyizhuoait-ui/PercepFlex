#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment A analysis: Equal-Budget Static vs Dynamic.

Compares (with REAL sliced-conv FLOPs):
  Static Tiny / Medium / Large        (forced widths on the dynamic checkpoint,
                                       same-weights operating points)
  Static Equal-Budget (independent)   (trained at width ~0.45, ~0.90G)
  Dynamic (free router)               (avg ~0.90G)
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from profiling.flops_real import count_flops  # noqa: E402
from models.static_model import StaticMultiTaskModel  # noqa: E402


def real_flops_table(model, x, enc_model=None):
    """Return {width: GFLOPs} for widths 0.25..1.0 and encoder-only."""
    out = {}
    if enc_model is not None:
        out["encoder"] = count_flops(enc_model, x) / 1e9
    for w in (0.25, 0.375, 0.5, 0.75, 1.0):
        model.set_width(w)
        out[w] = count_flops(model, x) / 1e9
    return out


def avg_dynamic_flops(alloc_csv, table):
    enc = table["encoder"]
    rows = list(csv.DictReader(open(alloc_csv)))
    wmap = {"0": 0.25, "1": 0.5, "2": 1.0}
    tot = enc
    for t in ("det", "da", "lane"):
        wsum = sum(wmap[r[t]] for r in rows) / len(rows)
        hf = np.interp(wsum, [0.25, 0.5, 1.0],
                       [(table[0.25] - enc) / 3, (table[0.5] - enc) / 3,
                        (table[1.0] - enc) / 3])
        tot += hf
    return tot


def main():
    import yaml
    import torch

    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "expA_equal_budget")
    with open(os.path.join(ROOT, "configs", "ours_static.yaml")) as f:
        cfg = yaml.safe_load(f)["model"]
    model = StaticMultiTaskModel(cfg).eval()
    enc = StaticMultiTaskModel(cfg).encoder.eval()
    x = torch.randn(1, 3, 640, 640)
    table = real_flops_table(model, x, enc)
    print("real FLOPs table (G):", {k: round(v, 3) for k, v in table.items()})

    # metric dirs to collect
    dirs = {
        "static_tiny": os.path.join(ROOT, "experiments", "exp_static_vs_dynamic", "static_tiny"),
        "static_medium": os.path.join(ROOT, "experiments", "exp_static_vs_dynamic", "static_medium"),
        "static_large": os.path.join(ROOT, "experiments", "exp_static_vs_dynamic", "static_large"),
        "dynamic": os.path.join(ROOT, "experiments", "exp_static_vs_dynamic", "dynamic"),
        "static_eb_s0": os.path.join(base, "static_eb_s0"),
    }
    rows = []
    for tag, d in dirs.items():
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            print(f"[{tag}] metrics missing")
            continue
        m = json.load(open(mp))
        # real flops: forced static widths / dynamic avg from allocation
        if tag == "static_tiny":
            fl = table[0.25]
        elif tag == "static_medium":
            fl = table[0.5]
        elif tag == "static_large":
            fl = table[1.0]
        elif tag == "dynamic":
            fl = avg_dynamic_flops(os.path.join(d, "allocation_statistics.csv"), table)
        else:
            fl = table[0.45] if 0.45 in table else table[0.5]
        rows.append({
            "model": tag, "flops_g": fl,
            "mAP50": m.get("mAP50", 0), "da_mIoU": m.get("da_mIoU", 0),
            "lane_fg_iou": m.get("lane_fg_iou", 0),
            "eval_fwd_ms": m.get("eval_fwd_ms", 0), "eval_fps": m.get("eval_fps", 0),
            "params": m.get("parameters", 0),
        })

    print(f"\n{'model':<16} {'FLOPs(G)':>9} {'mAP50':>7} {'DA_mIoU':>8} {'Lane_fgIoU':>10} {'fwd(ms)':>8} {'FPS':>7} {'Params(M)':>9}")
    for r in rows:
        print(f"{r['model']:<16} {r['flops_g']:>9.3f} {r['mAP50']:>7.4f} {r['da_mIoU']:>8.4f} "
              f"{r['lane_fg_iou']:>10.4f} {r['eval_fwd_ms']:>8.2f} {r['eval_fps']:>7.1f} {r['params']/1e6:>9.3f}")
    # case judgement
    dyn = next((r for r in rows if r["model"] == "dynamic"), None)
    seb = next((r for r in rows if r["model"] == "static_eb_s0"), None)
    if dyn and seb:
        print(f"\nCase judgement (dynamic {dyn['flops_g']:.3f}G vs static_eb {seb['flops_g']:.3f}G):")
        for k in ("mAP50", "da_mIoU", "lane_fg_iou"):
            d = dyn[k] - seb[k]
            print(f"  {k}: dynamic={dyn[k]:.4f} static_eb={seb[k]:.4f} delta={d:+.4f} "
                  f"({'Dyn>' if d > 0.005 else 'Dyn<' if d < -0.005 else 'Dyn≈'}Static)")


if __name__ == "__main__":
    main()
