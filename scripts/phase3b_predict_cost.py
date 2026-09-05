#!/usr/bin/env python
"""
Phase 3B · predict Params/FLOPs for the full 3x3 matrix BEFORE training.

Why: the cost columns are part of the deliverable, and a config typo (wrong
stem, z not actually applied) shows up here immediately instead of after
~14 GPU-hours. Also gives the expected values to cross-check the measured
numbers against once each cell finishes.
"""
import os
import sys

sys.path.insert(0, "/home/mycode/ai_study/trac")

import torch
import yaml

from models.static_model import StaticMultiTaskModel
from profiling.flops_real import count_flops


def build(cfg):
    # YAML nests the architecture under `model:` with input_size at top level;
    # StaticMultiTaskModel wants them flat. Unwrap rather than hand-roll a
    # second copy of the architecture (which could drift from the real one).
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def measure(cfg):
    m = build(cfg)
    m.eval()
    p = n_params(m)
    fl = None
    try:
        with torch.no_grad():
            fl = count_flops(m, torch.zeros(1, 3, 640, 640))
    except Exception:
        fl = None
    del m
    return p, fl


def main():
    root = "/home/mycode/ai_study/trac"
    encoders = ["esmall", "ebase", "elarge"]
    zs = [16, 32, 128]

    print(f"{'cell':<14} {'stem':>5} {'stages':<20} {'z':>4} {'params_M':>9} {'flops_G':>9}")
    print("-" * 66)
    prev = {}
    for enc in encoders:
        for z in zs:
            rel = (f"configs/phase3a_{enc}_z16.yaml" if z == 16
                   else f"configs/phase3b_{enc}_z{z}.yaml")
            cfg = yaml.safe_load(open(os.path.join(root, rel)))
            p, f = measure(cfg)
            pm = p / 1e6
            print(f"{enc}_z{z:<8} {cfg['model']['encoder']['stem']:>5} "
                  f"{str(cfg['model']['encoder']['stages']):<20} {z:>4} {pm:>9.4f} "
                  f"{(f/1e9 if f else float('nan')):>9.4f}")
            prev[(enc, z)] = pm
    print()
    print("monotonicity check (params must rise with z at fixed encoder):")
    for enc in encoders:
        vals = [prev[(enc, z)] for z in zs]
        ok = all(b > a for a, b in zip(vals, vals[1:]))
        print(f"  {enc:<8} {[f'{v:.4f}' for v in vals]}  {'OK' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
