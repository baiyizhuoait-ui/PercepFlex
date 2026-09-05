#!/usr/bin/env python3
"""Phase 3A · encoder-width scanner.

Goal: pick three encoder configs whose TOTAL model params (with z=16 and all
heads unchanged) land near 0.10M / 0.19M / 0.30M.

Constraints honoured:
  - ONLY encoder width changes (stem + stages channels).
  - blocks stay [2,2,2] (depth untouched).
  - representation z_channels stays 16 (never sacrificed to hit a budget).
  - detection / DA / lane heads untouched.

Widths are scaled multiplicatively from the baseline (16 / [32,64,96,128]) so
the encoder keeps its monotonically-widening shape instead of becoming an
arbitrary channel soup.
"""
import itertools, sys, os
sys.path.insert(0, "/home/mycode/ai_study/trac")

import torch
from models.static_model import StaticMultiTaskModel
from profiling.flops_real import count_flops

BASE_STEM = 16
BASE_STAGES = [32, 64, 96, 128]
TARGETS = {"E-small": 0.10e6, "E-base": 0.19e6, "E-large": 0.30e6}


def build(stem, stages, z=16):
    cfg = {
        "encoder": {"stem": stem, "stages": list(stages), "blocks": [2, 2, 2]},
        "representation": {"z_channels": z},
        "detection": {"nc": 1},
        "segmentation": {"hidden": 32},
        "input_size": [640, 640],
    }
    m = StaticMultiTaskModel(cfg)
    return m


def measure(stem, stages, z=16):
    m = build(stem, stages, z)
    p = sum(q.numel() for q in m.parameters())
    fl = None
    try:
        m.eval()
        with torch.no_grad():
            fl = count_flops(m, torch.zeros(1, 3, 640, 640))
    except Exception as e:
        fl = None
    del m
    return p, fl


def main():
    # Multiplicative width scale factors applied to BOTH stem and stages.
    scales = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
              0.95, 1.00, 1.10, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50,
              1.55, 1.60, 1.70, 1.80, 1.90, 2.00]
    rows = []
    print(f"{'scale':>6} {'stem':>5} {'stages':>22} {'params':>12} {'FLOPs_G':>9}")
    print("-" * 60)
    for s in scales:
        stem = max(4, int(round(BASE_STEM * s / 4.0) * 4))       # keep mult of 4
        stages = [max(4, int(round(c * s / 8.0) * 8)) for c in BASE_STAGES]
        p, fl = measure(stem, stages)
        rows.append({"scale": s, "stem": stem, "stages": stages,
                     "params": p, "flops": fl})
        fls = f"{fl/1e9:.3f}" if fl else "n/a"
        print(f"{s:>6.2f} {stem:>5} {str(stages):>22} {p/1e6:>11.4f}M {fls:>9}")

    print("\n=== closest to each target (total model params, z=16) ===")
    chosen = {}
    for name, tgt in TARGETS.items():
        best = min(rows, key=lambda r: abs(r["params"] - tgt))
        err = (best["params"] - tgt) / tgt * 100
        chosen[name] = best
        fls = f"{best['flops']/1e9:.3f}G" if best["flops"] else "n/a"
        print(f"  {name:<8} target={tgt/1e6:.3f}M  ->  scale={best['scale']:.2f}  "
              f"stem={best['stem']}  stages={best['stages']}  "
              f"params={best['params']/1e6:.4f}M ({err:+.1f}%)  flops={fls}")

    print("\n=== YAML snippets ===")
    for name, r in chosen.items():
        print(f"  # {name}: stem={r['stem']}, stages={r['stages']}, "
              f"params={r['params']/1e6:.4f}M")
        print(f"  encoder: {{stem: {r['stem']}, stages: {list(r['stages'])}, blocks: [2, 2, 2]}}")

    # sanity: the three must be strictly ordered in params and FLOPs
    ps = [chosen[k]["params"] for k in ("E-small", "E-base", "E-large")]
    fs = [chosen[k]["flops"] for k in ("E-small", "E-base", "E-large")]
    ok_p = ps[0] < ps[1] < ps[2]
    ok_f = all(f is not None for f in fs) and fs[0] < fs[1] < fs[2]
    print(f"\n  params strictly increasing: {ok_p}")
    print(f"  flops  strictly increasing: {ok_f}")
    if not (ok_p and ok_f):
        print("  [WARN] three tiers are not cleanly ordered — adjust the scale list")


if __name__ == "__main__":
    main()
