"""P5-STEP6 - lane causal intervention configs.

Three cells, all at Z=16 (the width where the shared-Z design is most
compressed, so any resolution effect is least confounded by spare capacity):

  l14up_z16   lane head runs at 1/4 on a bilinear upsample of Z, no new
              information. A learned upsampler. Params unchanged, head FLOPs
              x4 (~+0.53 GFLOPs).

  l14f1_z16   same, plus a 1x1 lateral from the encoder's 1/4 map. Adds the
              sub-cell position that Z discarded. +512 params.

  lch64_z16   lane head hidden 32 -> 64 at 1/8. Pure channel capacity, no
              resolution change. +32448 params, ~+0.41 GFLOPs.

The last cell exists so the comparison is Delta-metric / Delta-resource rather
than "high resolution helps", and its FLOP cost (+0.41) is deliberately close
to the resolution arm (+0.53). The question is not which is absolutely better,
it is which buys more lane per unit of compute.
"""
import copy
import os

import yaml

ROOT = "/home/mycode/ai_study/trac"
BASE = os.path.join(ROOT, "configs", "phase4a_r2_z16.yaml")

VARIANTS = {
    "l14up_z16": {"lane_res": 4, "lane_use_f1": False},
    "l14f1_z16": {"lane_res": 4, "lane_use_f1": True},
    "lch64_z16": {"lane_hidden": 64},
}

base = yaml.safe_load(open(BASE))
audit = []

for cell, seg_over in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["model"]["segmentation"] = dict(cfg["model"].get("segmentation", {}))
    cfg["model"]["segmentation"].update(seg_over)
    path = os.path.join(ROOT, "configs", "phase4b_%s.yaml" % cell)
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)

    # audit: the only difference from the R2 baseline must be in segmentation
    b = copy.deepcopy(base)
    diffs = []

    def walk(a, c, p=""):
        if isinstance(a, dict):
            for k in a:
                walk(a[k], c.get(k, {}) if isinstance(c, dict) else {}, p + "/" + str(k))
        else:
            if a != c:
                diffs.append("%s: %s -> %s" % (p, c, a))
    walk(cfg, b)
    audit.append((cell, path, seg_over, diffs))

print("Cells written, each differing from phase4a_r2_z16 ONLY where shown:")
for cell, path, over, diffs in audit:
    print("\n  %s  ->  %s" % (cell, os.path.basename(path)))
    print("    overrides: %s" % over)
    for d in diffs:
        print("    diff  %s" % d)

out = os.path.join(ROOT, "experiments", "phase5", "phase5_lane_probe_config_audit.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write("Phase 5 lane probe: configs vs phase4a_r2_z16 baseline\n")
    fh.write("=" * 70 + "\n")
    for cell, path, over, diffs in audit:
        fh.write("\n%s\n" % cell)
        fh.write("  overrides: %s\n" % over)
        for d in diffs:
            fh.write("  diff  %s\n" % d)
print("\nWROTE %s" % out)
