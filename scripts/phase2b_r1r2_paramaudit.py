"""Task 2(c) PRE-FLIGHT: is R1/R2 a fair comparison against R0?

Question this script answers
----------------------------
Under R0 the detection head DynamicDetHead reads the encoder multi-scale
features [F2(32ch,1/8), F3(64ch,1/16), F4(96ch,1/32)]. Under R1/R2 DetFromZ
reads Compact Z and builds its own 1/8 -> 1/16 -> 1/32 pyramid with a FIXED
internal width det_ch (default 32).

So R1/R2 differ from R0 in TWO ways at once:
  (1) INFORMATION SOURCE  - encoder features vs. one unified Z  (what we want to test)
  (2) HEAD CAPACITY       - per-scale input 32/64/96 vs. 32/32/32 (a confound)

If (2) is large, a detection drop under R1/R2 is NOT evidence about the
bottleneck; it is evidence about having a smaller head. This script measures
(2) so we can either control for it or report it honestly.

CPU only. No training, no dataset needed.
"""
import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from models.static_model import StaticMultiTaskModel  # noqa: E402

ENC = {"stem": 16, "stages": [32, 64, 96, 128], "blocks": [2, 2, 2]}


def build(zc, from_z, z_proj, det_ch=32):
    cfg = {
        "encoder": ENC,
        "representation": {"z_channels": zc},
        "detection": {"nc": 1, "from_z": from_z, "z_proj": z_proj, "det_ch": det_ch},
        "segmentation": {"hidden": 32},
    }
    return StaticMultiTaskModel(cfg)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", default="16,32,48,64,96,128,192,256")
    ap.add_argument("--det-ch", type=int, default=32)
    ap.add_argument("--out", default="experiments/phase2b/r1r2_paramaudit.csv")
    a = ap.parse_args()

    zs = [int(x) for x in a.zs.split(",")]
    rows = []
    for z in zs:
        for tag, (from_z, proj) in (("R0", (False, False)),
                                    ("R1", (True, False)),
                                    ("R2", (True, True))):
            m = build(z, from_z, proj, det_ch=a.det_ch)
            rows.append({
                "z": z,
                "variant": tag,
                "det_ch": a.det_ch,
                "params_total": nparams(m),
                "params_encoder": nparams(m.encoder),
                "params_repr": nparams(m.representation),
                "params_det": nparams(m.det_head),
                "params_da": nparams(m.da_head),
                "params_lane": nparams(m.lane_head),
            })
            del m

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("%4s | %8s %8s %7s | %8s %8s %8s" % ("z", "R0 det", "R2 det", "R2/R0", "R0 tot", "R2 tot", "dTot"))
    print("-" * 62)
    by = {(r["z"], r["variant"]): r for r in rows}
    for z in zs:
        r0, r2 = by[(z, "R0")], by[(z, "R2")]
        print("%4d | %8d %8d %7.3f | %8d %8d %+8d" % (
            z, r0["params_det"], r2["params_det"],
            r2["params_det"] / r0["params_det"],
            r0["params_total"], r2["params_total"],
            r2["params_total"] - r0["params_total"]))

    print("")
    print("R2 det-head params across the Z sweep (flat == det_ch isolates head capacity):")
    r2s = [by[(z, "R2")]["params_det"] for z in zs]
    print("  min=%d  max=%d  spread=%d" % (min(r2s), max(r2s), max(r2s) - min(r2s)))

    print("")
    print("Wrote " + a.out)


if __name__ == "__main__":
    main()
