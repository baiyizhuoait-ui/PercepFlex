#!/usr/bin/env python
"""Phase 4A - generate the three R2 configs and prove they differ from R0 in ONE place.

Each R2 config is produced from the R0 config that already ran at the same
encoder and z, by adding detection.from_z / z_proj / det_ch. A flatten-and-diff
is then run so the claim "only the information path changed" is checked rather
than asserted.
"""
import os, sys, copy
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments/phase4a")
AUDIT = os.path.join(OUT, "phase4A_config_audit.txt")
os.makedirs(OUT, exist_ok=True)

TPL = {16: "configs/phase3a_ebase_z16.yaml",
       32: "configs/phase3b_ebase_z32.yaml",
       128: "configs/phase3b_ebase_z128.yaml"}

w = []
def p(s=""):
    w.append(s); print(s)

def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = "%s%s" % (pre, k)
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, list):
            out[key] = tuple(v)
        else:
            out[key] = v
    return out

def same(a, b):
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return False

p("=" * 100)
p("PHASE 4A - R2 CONFIG GENERATION + AUDIT")
p("=" * 100)
p("each R2 config is derived from the R0 config that already ran at the same")
p("encoder and z; the ONLY intended change is the detection information path")
p()

allok = True
for z, rel in TPL.items():
    src = os.path.join(ROOT, rel)
    cfg0 = yaml.safe_load(open(src))
    cfg2 = copy.deepcopy(cfg0)
    det = cfg2["model"].setdefault("detection", {})
    det["from_z"] = True
    det["z_proj"] = True
    det["det_ch"] = 32
    path = os.path.join(ROOT, "configs/phase4a_r2_z%d.yaml" % z)
    with open(path, "w") as f:
        yaml.safe_dump(cfg2, f, default_flow_style=False, sort_keys=False)

    back = yaml.safe_load(open(path))
    f0, f2 = flat(cfg0), flat(back)
    diff = sorted(k for k in set(f0) | set(f2) if not same(f0.get(k), f2.get(k)))
    allowed = {"model.detection.from_z", "model.detection.z_proj", "model.detection.det_ch"}
    bad = [k for k in diff if k not in allowed]
    p("-" * 100)
    p("  z=%d   %s -> %s" % (z, rel, os.path.relpath(path, ROOT)))
    p("    fields changed      : %s" % (", ".join(diff) if diff else "NONE (suspect)"))
    p("    unexpected changes  : %s" % (", ".join(bad) if bad else "none"))
    p("    z_channels          : %s   (unchanged)" % back["model"]["representation"]["z_channels"])
    p("    encoder             : stem %s  stages %s  blocks %s  (unchanged)" % (
        back["model"]["encoder"]["stem"], back["model"]["encoder"]["stages"],
        back["model"]["encoder"]["blocks"]))
    p("    epochs / bs / lr    : %s / %s / %s   (unchanged)" % (
        back["train"]["epochs"], back["train"]["batch_size"], back["train"]["lr"]))
    ok = (not bad) and bool(diff)
    p("    VERDICT: %s" % ("PASS" if ok else "FAIL"))
    p()
    allok = allok and ok

p("=" * 100)
p("R2 CONFIG AUDIT: %s" % ("PASS - only the detection information path differs"
                            if allok else "FAIL"))
p("=" * 100)
open(AUDIT, "w").write("\n".join(w) + "\n")
print()
print("written:", AUDIT)
sys.exit(0 if allok else 1)
