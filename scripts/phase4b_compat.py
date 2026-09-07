#!/usr/bin/env python
"""Phase 4A Level 2 probe A - model-side compatibility and cost check.

Two questions, in this order:

1. BACKWARD COMPATIBILITY. With `task_proj` absent from the config, R0 and R2
   must build bit-identically to before the change. Checked against the numbers
   already committed in experiments/phase4a/phase4A_cost_breakdown.csv, not
   against a number retyped here.

2. COST OF THE PROBE. What R3 (per-task projection) actually adds, per cell, so
   the probe result can be read with the confound visible.

Also prints, for each R3 cell, whether each task projection is a real conv or an
Identity no-op - that determines which arms can move at all and is needed to
read the result honestly.
"""
import os, sys, copy
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml, torch
from models.static_model import StaticMultiTaskModel

ROOT = "/home/mycode/ai_study/trac"
COST = os.path.join(ROOT, "experiments/phase4a/phase4A_cost_breakdown.csv")

TPL = {16: "configs/phase3a_ebase_z16.yaml",
       32: "configs/phase3b_ebase_z32.yaml",
       128: "configs/phase3b_ebase_z128.yaml"}

# Fixed absolute targets, identical at every shared-Z width (probe design rule 1).
TARGETS = {"det": 32, "lane": 32, "da": 16}

torch.manual_seed(0)


def load(z):
    with open(os.path.join(ROOT, TPL[z])) as fh:
        return yaml.safe_load(fh)


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def to_r2(cfg):
    c = copy.deepcopy(cfg)
    c["model"]["detection"]["from_z"] = True
    c["model"]["detection"]["z_proj"] = True
    c["model"]["detection"]["det_ch"] = 32
    return c


def to_r3(cfg, targets=TARGETS):
    c = to_r2(cfg)
    c["model"]["task_proj"] = dict(enabled=True, **targets)
    return c


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def parts(m):
    return {
        "encoder": nparams(m.encoder),
        "representation": nparams(m.representation),
        "det_head": nparams(m.det_head),
        "da_head": nparams(m.da_head) + (nparams(m.da_proj) if m.task_proj else 0),
        "lane_head": nparams(m.lane_head) + (nparams(m.lane_proj) if m.task_proj else 0),
    }


def committed():
    """Read the already-committed cost table, tolerating the unquoted tuple."""
    out = {}
    with open(COST) as fh:
        head = None
        for ln in fh:
            f, buf, d = [], "", 0
            for ch in ln.rstrip("\n"):
                if ch == "(":
                    d += 1
                elif ch == ")":
                    d -= 1
                if ch == "," and d == 0:
                    f.append(buf)
                    buf = ""
                else:
                    buf += ch
            f.append(buf)
            f = [x.strip() for x in f]
            if head is None:
                head = f
                continue
            r = dict(zip(head, f))
            out[(r["variant"], int(r["z"]))] = r
    return out


com = committed()

print("=" * 74)
print("1. BACKWARD COMPATIBILITY (task_proj absent -> must reproduce committed R0/R2)")
print("=" * 74)
ok = True
for z in (16, 32, 128):
    for tag, fn in (("R0", lambda c: copy.deepcopy(c)), ("R2", to_r2)):
        m = build(fn(load(z)))
        p = parts(m)
        tot = sum(p.values())
        ref = int(com[(tag, z)]["params_total"])
        match = (tot == ref)
        ok = ok and match
        print("  z=%-4d %s  total=%d  committed=%d  %s"
              % (z, tag, tot, ref, "OK" if match else "MISMATCH"))
        if not match:
            for k in ("encoder", "representation", "det_head", "da_head", "lane_head"):
                print("      %-14s now=%d committed=%s"
                      % (k, p[k], com[(tag, z)]["params_%s" % k]))
        assert not m.task_proj, "task_proj must stay off when absent from config"
print("  -> all identical:", ok)

print()
print("=" * 74)
print("2. R3 COST AND WHICH PROJECTIONS ARE REAL")
print("=" * 74)
for z in (16, 32):
    m2 = build(to_r2(load(z)))
    m3 = build(to_r3(load(z)))
    p2, p3 = parts(m2), parts(m3)
    t2, t3 = sum(p2.values()), sum(p3.values())
    print("  z=%d  targets det=%d lane=%d da=%d"
          % (z, TARGETS["det"], TARGETS["lane"], TARGETS["da"]))
    print("    da_proj   : %s" % m3.da_proj.extra_repr())
    print("    lane_proj : %s" % m3.lane_proj.extra_repr())
    print("    det_head  : %d -> %d  (%+d)" % (p2["det_head"], p3["det_head"],
                                               p3["det_head"] - p2["det_head"]))
    print("    da_head   : %d -> %d  (%+d)" % (p2["da_head"], p3["da_head"],
                                               p3["da_head"] - p2["da_head"]))
    print("    lane_head : %d -> %d  (%+d)" % (p2["lane_head"], p3["lane_head"],
                                               p3["lane_head"] - p2["lane_head"]))
    print("    total     : %d -> %d  (%+d, %+.2f%%)"
          % (t2, t3, t3 - t2, 100.0 * (t3 - t2) / t2))
    m3.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 640, 640)
        det, da, lane, zz = m3(x, return_z=True)
    print("    shapes    : z=%s det=%s da=%s lane=%s"
          % (tuple(zz.shape), tuple(det.shape), tuple(da.shape), tuple(lane.shape)))
    print()

print("=" * 74)
print("3. WHAT EACH ARM CAN ACTUALLY TEST (pre-registered)")
print("=" * 74)
print("  z16: det already projects 16->32 (unchanged), lane 16->32 is the ONLY")
print("       real change, da 16->16 is Identity. -> predicts lane moves, det/da flat.")
print("  z32: det 32->32 and lane 32->32 are Identity, da 32->16 is the ONLY real")
print("       change. -> predicts da flat (it is z-indifferent), det/lane flat.")
print("  Cross-arm: if task-specific projection decouples task capacity from the")
print("       shared width, then R3-z32 minus R3-z16 should be SMALLER than")
print("       R2-z32 minus R2-z16 on detection and lane.")
