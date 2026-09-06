#!/usr/bin/env python
"""Phase 3C - design the two new intermediate encoders used ONLY for budget matching.

Goal: one encoder between E-small and E-base, one between E-base and E-large,
each paired with z=32, such that total params land on the Budget-L (0.19M) and
Budget-M (0.29M) targets.

Hard constraints (Phase 3C sec.8):
  * width interpolation ONLY along the existing stem/stages scaling law
  * blocks [2,2,2] untouched, depth untouched, no new modules
  * stages rounded to multiples of 8, same as every existing encoder
"""
import os, sys, copy
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml, torch
from models.static_model import StaticMultiTaskModel

ROOT = "/home/mycode/ai_study/trac"
TPL = os.path.join(ROOT, "configs/phase3b_ebase_z32.yaml")

KNOWN = {
    "esmall": {"stem": 12, "stages": [24, 40, 64, 80]},
    "ebase":  {"stem": 16, "stages": [32, 64, 96, 128]},
    "elarge": {"stem": 20, "stages": [40, 80, 128, 168]},
}
TARGETS = [("midL", "esmall", "ebase", 0.19, 32),
           ("midM", "ebase", "elarge", 0.29, 32)]

def build(cfg):
    flat = dict(cfg["model"]); flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)

def nparams(cfg):
    m = build(cfg); n = sum(p.numel() for p in m.parameters()); del m
    return n / 1e6

def r8(x): return int(round(x / 8.0) * 8)

def interp(a, b, t):
    stem = int(round(a["stem"] + t * (b["stem"] - a["stem"])))
    stages = [r8(x + t * (y - x)) for x, y in zip(a["stages"], b["stages"])]
    return stem, stages

def main():
    base = yaml.safe_load(open(TPL))
    assert base["model"]["representation"]["z_channels"] == 32
    out = []
    print("=" * 96)
    print("PHASE 3C - NEW ENCODER SEARCH (budget matching only, z=32)")
    print("=" * 96)
    for tag, ea, eb, target, z in TARGETS:
        A, B = KNOWN[ea], KNOWN[eb]
        print()
        print("  %s: interpolate %s -> %s, target %.4f M at z=%d" % (tag, ea, eb, target, z))
        print("    %-8s %-22s" % ("t", "stages"))
        seen, cands = set(), []
        for i in range(0, 201):
            t = i / 200.0
            if t < 0.05 or t > 0.995: continue
            stem, stages = interp(A, B, t)
            if not all(stages[j] < stages[j + 1] for j in range(3)): continue
            key = (stem, tuple(stages))
            if key in seen: continue
            seen.add(key)
            cfg = copy.deepcopy(base)
            cfg["model"]["encoder"]["stem"] = stem
            cfg["model"]["encoder"]["stages"] = stages
            cfg["model"]["representation"]["z_channels"] = z
            pm = nparams(cfg)
            cands.append((abs(pm - target), t, stem, stages, pm))
        cands.sort(key=lambda c: c[0])
        for dev, t, stem, stages, pm in cands[:5]:
            print("    %-8.3f %-22s stem %-3d params %.4f  dev %+.4f (%+.2f%%)" % (
                t, str(stages), stem, pm, pm - target, (pm - target) / target * 100))
        dev, t, stem, stages, pm = cands[0]
        out.append(dict(tag=tag, stem=stem, stages=stages, z=z, t=t,
                        params=pm, target=target,
                        dev_pct=(pm - target) / target * 100,
                        parent_a=ea, parent_b=eb))
        print("    -> CHOSEN t=%.3f stem=%d stages=%s  params %.4f  dev %+.2f%%" % (
            t, stem, stages, pm, (pm - target) / target * 100))
    print()
    print("-" * 96)
    print("  summary")
    print("-" * 96)
    for o in out:
        print("  %-6s stem %-3d stages %-22s z%-4d params %.4f  target %.2f  dev %+.2f%%" % (
            o["tag"], o["stem"], str(o["stages"]), o["z"], o["params"], o["target"], o["dev_pct"]))
    with open(os.path.join(ROOT, "experiments/phase3c/phase3C_new_encoders.txt"), "w") as f:
        for o in out:
            f.write("%s\t%d\t%s\t%d\t%.6f\t%.2f\t%+.2f\n" % (
                o["tag"], o["stem"], ",".join(map(str, o["stages"])), o["z"],
                o["params"], o["target"], o["dev_pct"]))
    print()
    print("written: experiments/phase3c/phase3C_new_encoders.txt")
    return 0

if __name__ == "__main__":
    sys.exit(main())
