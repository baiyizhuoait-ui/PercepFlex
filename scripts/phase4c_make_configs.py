#!/usr/bin/env python
"""Level 2 probe B - reconstruction configs, audit, and cost.

Probe B asks whether the reconstruction is what limits detection on a shared Z.
YOLOF (CVPR 2021) diagnosed the single-level-feature failure as a limited
receptive field; our R2 reconstruction is a single 1x1, i.e. a one-pixel
receptive field at 1/8 resolution. See docs/PHASE4B_LITERATURE_SYNTHESIS.md.

Three cells at 4 epochs, seed 0, E-base:

    r2d_z16   R2 + dilated reconstruction (2 depthwise residual blocks, rates 2,4)
    r2d_z32   same at shared Z=32
    r2p_z16   R2 + identical blocks at dilation 1   <- CONTROL

The control is what makes this readable: r2p_z16 has exactly the same parameters
and the same depth as r2d_z16 and differs only in dilation. If r2d beats r2p,
the gain is receptive field. If they are equal, the gain (if any) was just depth,
and the reconstruction story is about capacity rather than scale range.

NOTE: this script builds models to count parameters. It runs on CPU on purpose -
it must not touch the GPU while a training job is running.
"""
import os, sys, copy
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml, torch
from models.static_model import StaticMultiTaskModel

ROOT = "/home/mycode/ai_study/trac"
AUDIT = os.path.join(ROOT, "experiments/phase4a/phase4C_config_audit.txt")

BASE = {"r2d_z16": ("configs/phase4a_r2_z16.yaml", "dw_dilated"),
        "r2d_z32": ("configs/phase4a_r2_z32.yaml", "dw_dilated"),
        "r2p_z16": ("configs/phase4a_r2_z16.yaml", "dw_plain")}

ALLOWED = {"model.detection.rec", "train.epochs"}

torch.manual_seed(0)


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


def build(cfg):
    f = dict(cfg["model"])
    f["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(f)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


lines = []
def p(s=""):
    lines.append(s)
    print(s)


p("Level 2 probe B config audit")
p("=" * 72)
p("")

ok_all = True
for cell, (base, rec) in BASE.items():
    b = load(base)
    c = copy.deepcopy(b)
    c["train"]["epochs"] = 4
    c["model"]["detection"]["rec"] = rec

    fb, fc = flat(b), flat(c)
    keys = sorted(set(fb) | set(fc))
    diffs = [(k, fb.get(k, "<absent>"), fc.get(k, "<absent>"))
             for k in keys if fb.get(k, "<absent>") != fc.get(k, "<absent>")]
    bad = [d for d in diffs if d[0] not in ALLOWED]
    ok = not bad
    ok_all = ok_all and ok

    p("%-9s base=%s  rec=%s" % (cell, base, rec))
    for k, a, bb in diffs:
        p("    [%s] %-28s %s -> %s" % ("OK " if k in ALLOWED else "BAD", k, a, bb))
    p("    verdict: %s" % ("PASS" if ok else "FAIL - unexpected key changed"))

    with open(os.path.join(ROOT, "configs/phase4b_%s.yaml" % cell), "w") as fh:
        yaml.safe_dump(c, fh, sort_keys=False)

    m = build(c)
    tot = nparams(m)
    det = nparams(m.det_head)
    if rec == "1x1":
        extra = ""
    else:
        rates = [blk.f[0].dilation[0] for blk in m.det_head.rec_op]
        extra = "  rates=%s" % (rates,)
    p("    params: total=%d  det_head=%d%s" % (tot, det, extra))
    p("")

# baseline for the delta
base16 = build(load("configs/phase4a_r2_z16.yaml"))
base32 = build(load("configs/phase4a_r2_z32.yaml"))
p("delta vs the R2 baseline it is compared against:")
for cell, ref, tag in (("r2d_z16", base16, "r2u_z16"),
                       ("r2p_z16", base16, "r2u_z16"),
                       ("r2d_z32", base32, "r2u_z32")):
    m = build(load("configs/phase4b_%s.yaml" % cell))
    d = nparams(m) - nparams(ref)
    p("  %-9s vs %s : %+d params (%+.2f%%)" % (cell, tag, d, 100.0 * d / nparams(ref)))
p("")
p("r2d_z16 and r2p_z16 differ ONLY in dilation: %s"
  % (nparams(build(load("configs/phase4b_r2d_z16.yaml")))
     == nparams(build(load("configs/phase4b_r2p_z16.yaml")))))
p("")
p("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))

with open(AUDIT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
sys.exit(0 if ok_all else 1)
