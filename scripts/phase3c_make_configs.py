#!/usr/bin/env python
"""Phase 3C - generate the two budget-matching configs + static audit.

Only model.encoder.stem and model.encoder.stages may differ from the Phase 3B
z=32 template. Everything else must be byte-equivalent in VALUE, which is
enforced here by a flatten-and-diff, not by eyeballing the file.
"""
import os, sys, copy
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml
from models.static_model import StaticMultiTaskModel

ROOT = "/home/mycode/ai_study/trac"
TPL = os.path.join(ROOT, "configs/phase3b_ebase_z32.yaml")
REF3A = os.path.join(ROOT, "configs/phase3a_ebase_z16.yaml")
AUDIT = os.path.join(ROOT, "experiments/phase3c/phase3C_config_audit.txt")

NEW = [("midL", 15, [32, 64, 88, 120], 0.19),
       ("midM", 19, [40, 80, 120, 160], 0.29)]

def build(cfg):
    flat = dict(cfg["model"]); flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)

def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict): out.update(flat(v, key + "."))
        elif isinstance(v, list): out[key] = tuple(v)
        else: out[key] = v
    return out

def same(a, b):
    if a == b: return True
    try: return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError): return False

w = []
def p(s=""):
    w.append(s); print(s)

p("=" * 100)
p("PHASE 3C - CONFIG GENERATION + STATIC AUDIT")
p("=" * 100)
p("template : %s" % os.path.relpath(TPL, ROOT))
p("reference: %s" % os.path.relpath(REF3A, ROOT))
p("allowed to change: model.encoder.stem, model.encoder.stages  (nothing else)")
p()

tpl = yaml.safe_load(open(TPL))
ref = yaml.safe_load(open(REF3A))
ftpl, fref = flat(tpl), flat(ref)

p("-" * 100)
p("### LAYER 0 - template already matches the Phase 3A protocol")
p("-" * 100)
d0 = [k for k in set(ftpl) | set(fref) if not same(ftpl.get(k), fref.get(k))]
d0 = [k for k in d0 if "encoder" not in k and "z_channels" not in k]
p("  non-encoder differences vs Phase 3A ebase_z16: %d" % len(d0))
for k in sorted(d0):
    p("    %s: %r -> %r" % (k, fref.get(k), ftpl.get(k)))
p("  -> %s" % ("PASS (protocol untouched)" if not d0 else "FAIL"))
p()

allok = not d0
for tag, stem, stages, target in NEW:
    cfg = copy.deepcopy(tpl)
    cfg["model"]["encoder"]["stem"] = stem
    cfg["model"]["encoder"]["stages"] = list(stages)
    path = os.path.join(ROOT, "configs/phase3c_%s_z32.yaml" % tag)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    written = yaml.safe_load(open(path))
    p("-" * 100)
    p("### %s  -> %s" % (tag, os.path.relpath(path, ROOT)))
    p("-" * 100)
    p("  stem=%d  stages=%s  z=%s  blocks=%s  input_size=%s" % (
        written["model"]["encoder"]["stem"], written["model"]["encoder"]["stages"],
        written["model"]["representation"]["z_channels"],
        written["model"]["encoder"].get("blocks"), written.get("input_size")))

    fw = flat(written)
    diff = [k for k in set(fw) | set(ftpl) if not same(fw.get(k), ftpl.get(k))]
    allowed = {"model.encoder.stem", "model.encoder.stages"}
    bad = [k for k in diff if k not in allowed]
    p("  fields differing from template: %s" % (sorted(diff) if diff else "none"))
    p("  unexpected differences        : %s" % (sorted(bad) if bad else "none"))
    ok1 = not bad

    m = build(written)
    npar = sum(q.numel() for q in m.parameters()) / 1e6
    del m
    dev = (npar - target) / target * 100
    p("  params %.4f M   target %.2f M   dev %+.2f%%" % (npar, target, dev))
    ok2 = abs(dev) <= 3.0

    ok3 = (written["model"]["encoder"]["blocks"] == [2, 2, 2]
           and written["model"]["representation"]["z_channels"] == 32
           and list(written["model"]["encoder"]["stages"]) == list(stages))
    p("  invariants (blocks [2,2,2], z=32, stages applied): %s" % ("PASS" if ok3 else "FAIL"))
    p("  CELL VERDICT: %s" % ("PASS" if (ok1 and ok2 and ok3) else "FAIL"))
    p()
    allok = allok and ok1 and ok2 and ok3

p("=" * 100)
p("STATIC AUDIT: %s" % ("PASS - only encoder width differs; protocol identical" if allok else "FAIL"))
p("=" * 100)
os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
open(AUDIT, "w").write("\n".join(w) + "\n")
print(); print("written:", AUDIT)
sys.exit(0 if allok else 1)
