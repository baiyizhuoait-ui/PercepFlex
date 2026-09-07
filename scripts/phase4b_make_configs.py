#!/usr/bin/env python
"""Phase 4A Level 2 probe A - config generation + audit.

Four cells at 4 epochs, seed 0, E-base:

    r2u_z16   R2, shared Z=16, no per-task projection   (REUSED from STEP 2 sanity)
    r2u_z32   R2, shared Z=32, no per-task projection
    r3tp_z16  R3, shared Z=16 + per-task projection
    r3tp_z32  R3, shared Z=32 + per-task projection

R3 targets are FIXED ABSOLUTE WIDTHS {det: 32, lane: 32, da: 16}, identical at
both shared-Z widths. That is the point of the probe: across the two R3 arms
every head sees exactly the same input width, so only the shared Z changes.

The audit flattens each generated config and diffs it against its R2 base, then
asserts that nothing outside an explicit allow-list moved. This is the same
discipline as scripts/phase4a_make_configs.py and exists so that a later reader
can be sure the probe differs from R2 in one respect only.
"""
import os, copy, sys
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml

ROOT = "/home/mycode/ai_study/trac"
OUTCFG = os.path.join(ROOT, "configs")
AUDIT = os.path.join(ROOT, "experiments/phase4a/phase4B_config_audit.txt")

# Fixed absolute per-task projection targets (probe design rule 1).
TARGETS = {"det": 32, "lane": 32, "da": 16}

BASE = {"r2u_z16": "configs/phase4a_r2_z16.yaml",
        "r2u_z32": "configs/phase4a_r2_z32.yaml",
        "r3tp_z16": "configs/phase4a_r2_z16.yaml",
        "r3tp_z32": "configs/phase4a_r2_z32.yaml"}

ALLOWED = {"model.task_proj.enabled", "model.task_proj.det",
           "model.task_proj.lane", "model.task_proj.da",
           "train.epochs"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = "%s%s" % (pre, k)
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def load(p):
    with open(os.path.join(ROOT, p)) as fh:
        return yaml.safe_load(fh)


lines = []
def p(s=""):
    lines.append(s)
    print(s)


p("Phase 4B (Level 2 probe A) config audit")
p("=" * 70)
p("R3 per-task projection targets: det=%d lane=%d da=%d (absolute, same at both z)"
  % (TARGETS["det"], TARGETS["lane"], TARGETS["da"]))
p("")

ok_all = True
for cell, base in BASE.items():
    b = load(base)
    c = copy.deepcopy(b)
    c["train"]["epochs"] = 4
    if cell.startswith("r3tp"):
        c["model"]["task_proj"] = dict(enabled=True, **TARGETS)

    # ---- audit against the R2 base ----
    fb, fc = flat(b), flat(c)
    keys = sorted(set(fb) | set(fc))
    diffs = [(k, fb.get(k, "<absent>"), fc.get(k, "<absent>"))
             for k in keys if fb.get(k, "<absent>") != fc.get(k, "<absent>")]
    bad = [d for d in diffs if d[0] not in ALLOWED]
    ok = not bad
    ok_all = ok_all and ok

    p("%-9s base=%s" % (cell, base))
    p("  differences vs base (%d):" % len(diffs))
    for k, a, bb in diffs:
        flag = "OK " if k in ALLOWED else "BAD"
        p("    [%s] %-28s %s -> %s" % (flag, k, a, bb))
    p("  verdict: %s" % ("PASS - only allow-listed keys moved" if ok
                         else "FAIL - unexpected key changed"))
    p("")

    with open(os.path.join(OUTCFG, "phase4b_%s.yaml" % cell), "w") as fh:
        yaml.safe_dump(c, fh, sort_keys=False)

p("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))
p("")
p("Note on r2u_z16: its 4ep result already exists from Phase 4A STEP 2 sanity")
p("(experiments/phase4a/exp4A_r2_z16_e4, commit 63fa17b). It is REUSED, not")
p("retrained: the model builds to the identical 201366 parameters (verified by")
p("scripts/phase4b_compat.py) and task_proj defaults off, so the forward graph is")
p("unchanged. Its provenance is recorded as source=reused:phase4a-sanity.")

with open(AUDIT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
sys.exit(0 if ok_all else 1)
