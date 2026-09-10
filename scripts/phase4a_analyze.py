#!/usr/bin/env python
"""P4A-STEP4 - R0 vs R2 paired analysis.

The central question: does detection become Z-sensitive once it is forced
through Z? That is answered by comparing the SAME z-contrast inside R0 and
inside R2 at a pinned encoder (E-base), not by comparing absolute scores.

R0 rows are reused from the CSVs that already exist; R0 is never retrained.
"""
import csv, os, sys

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments/phase4a")
PAIRED = os.path.join(OUT, "phase4A_r0_vs_r2.csv")
ANALYSIS = os.path.join(OUT, "phase4A_analysis.txt")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
TASK_OF = {"mAP50": "detection", "mAP50_95": "detection",
           "da_mIoU": "DA", "da_fg": "DA",
           "lane_mIoU": "lane", "lane_fg": "lane"}
TASKS = ["detection", "DA", "lane"]
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}
ZS = [16, 32, 128]
ENC = "ebase"

w = []
def p(s=""):
    w.append(s); print(s)

def num(r, k):
    try:
        v = float(r[k])
        return v if v == v else None
    except (KeyError, TypeError, ValueError):
        return None

def load_r0():
    rows = {}
    for rel in ("experiments/phase3a/exp3A_encoder.csv",
                "experiments/phase3b/phase3B_encoder_z.csv"):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r.get("encoder") != ENC:
                continue
            try:
                z = int(r["z"])
            except (KeyError, ValueError):
                continue
            if z in ZS and z not in rows:
                rows[z] = r
    return rows

def load_r2():
    rows = {}
    path = os.path.join(ROOT, "experiments/phase4a/phase4A_results.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if str(r.get("epochs")) != "20":
                continue
            try:
                z = int(r["z"])
            except (KeyError, ValueError):
                continue
            rows[z] = r
    return rows

r0, r2 = load_r0(), load_r2()

p("=" * 100)
p("PHASE 4A · R0 vs R2 PAIRED ANALYSIS")
p("=" * 100)
p("encoder pinned: %s    protocol: 20ep / seed 0 / bs 16 / lr 1e-3 / 640x640" % ENC)
p("R0 rows reused from existing CSVs; R0 is never retrained.")
p("Noise floor is an EXTERNAL reference (Phase 2-C, 3 seeds, 4 epochs).")
p("Phase 4A is single-seed and does NOT estimate its own variance.")
p()
p("R0 cells found: %s" % ", ".join("z%d" % z for z in sorted(r0)) or "none")
p("R2 cells found: %s" % ", ".join("z%d" % z for z in sorted(r2)) or "none")
p()

missing = [z for z in ZS if z not in r2]
if missing:
    p("!! R2 is missing z=%s - the main experiment is not finished." % missing)
    p("   Everything below is partial and must not be used for the stopping rule.")
    p()

# ---------------------------------------------------------------- A
p("-" * 100)
p("### A. SIDE BY SIDE AT EQUAL z")
p("-" * 100)
p()
p("  %-5s %-4s %8s %8s   %s" % ("z", "var", "params", "flops", "  ".join("%9s" % m for m in METRICS)))
for z in ZS:
    for tag, src in (("R0", r0), ("R2", r2)):
        r = src.get(z)
        if not r:
            p("  %-5d %-4s %8s %8s   (missing)" % (z, tag, "-", "-"))
            continue
        vals = []
        for m in METRICS:
            v = num(r, m)
            vals.append("%9.4f" % v if v is not None else "%9s" % "NA")
        p("  %-5d %-4s %8s %8s   %s" % (
            z, tag,
            ("%.4f" % num(r, "params_M")) if num(r, "params_M") else "NA",
            ("%.4f" % num(r, "flops_G")) if num(r, "flops_G") else "NA",
            "  ".join(vals)))
p()

# ---------------------------------------------------------------- B
p("-" * 100)
p("### B. R2 - R0 AT EQUAL z  (cost of forcing detection through Z)")
p("-" * 100)
p()
p("  %-5s %10s %10s   %s" % ("z", "dParams", "dFLOPs", "  ".join("%10s" % m for m in METRICS)))
for z in ZS:
    a, b = r0.get(z), r2.get(z)
    if not (a and b):
        continue
    dp = (num(b, "params_M") or 0) - (num(a, "params_M") or 0)
    df = (num(b, "flops_G") or 0) - (num(a, "flops_G") or 0)
    cells = []
    for m in METRICS:
        va, vb = num(a, m), num(b, m)
        if va is None or vb is None:
            cells.append("%10s" % "NA")
        else:
            d = vb - va
            flag = "*" if abs(d) > NOISE[m] else " "
            cells.append("%+9.4f%s" % (d, flag))
    p("  %-5d %+10.4f %+10.4f   %s" % (z, dp, df, "  ".join(cells)))
p("  (* = beyond the noise floor)")
p()

# ---------------------------------------------------------------- C : the key test
p("-" * 100)
p("### C. Z-SENSITIVITY: THE SAME CONTRAST INSIDE R0 AND INSIDE R2")
p("-" * 100)
p()
p("  This is the test for 'bypass masking'. We do not compare absolute scores;")
p("  we compare how much each model MOVES when z goes 16 -> 32 -> 128.")
p()
p("  %-10s %-6s %12s %12s %12s" % ("metric", "model", "z16->z32", "z16->z128", "noise"))
sens = {}
for m in METRICS:
    for tag, src in (("R0", r0), ("R2", r2)):
        v16 = num(src.get(16, {}), m)
        v32 = num(src.get(32, {}), m)
        v128 = num(src.get(128, {}), m)
        if None in (v16, v32, v128):
            continue
        d32, d128 = v32 - v16, v128 - v16
        sens[(m, tag)] = (d32, d128)
        p("  %-10s %-6s %+12.4f %+12.4f %12.4f" % (m, tag, d32, d128, NOISE[m]))
    p()
p()
p("  reading it as multiples of the noise floor:")
p()
p("  %-10s %-6s %14s %14s" % ("metric", "model", "|z16->z32|/n", "|z16->z128|/n"))
for m in METRICS:
    for tag in ("R0", "R2"):
        if (m, tag) not in sens:
            continue
        d32, d128 = sens[(m, tag)]
        p("  %-10s %-6s %14.2f %14.2f" % (m, tag, abs(d32) / NOISE[m], abs(d128) / NOISE[m]))
    p()

# ---------------------------------------------------------------- D : verdict per task
p("-" * 100)
p("### D. DID DETECTION BECOME Z-SENSITIVE?  (H-01 / H-02)")
p("-" * 100)
p()
det = {}
for tag in ("R0", "R2"):
    ms = [m for m in METRICS if TASK_OF[m] == "detection"]
    worst = 0.0
    for m in ms:
        if (m, tag) in sens:
            worst = max(worst, abs(sens[(m, tag)][1]) / NOISE[m])
    det[tag] = worst
    p("  %s  largest detection |z16->z128| = %.2f x noise" % (tag, worst))
p()
if det.get("R2", 0) > 1.0 and det.get("R2", 0) > det.get("R0", 0):
    p("  ==> H-01 SUPPORTED: detection is more Z-sensitive in R2 than in R0.")
    p("      The earlier 'Z does not matter for detection' was at least partly")
    p("      the R0 bypass, not a property of detection itself.")
elif det.get("R2", 0) <= 1.0:
    p("  ==> H-02 NOT SUPPORTED at this encoder: even with the bypass removed,")
    p("      detection stays inside the noise floor across z16/32/128.")
    p("      Do NOT conclude 'Z is useless' - see the explanation list below.")
    p("      The designated next test for explanation C (encoder is the binding")
    p("      constraint) is one R2 probe at E-large.")
else:
    p("  ==> inconclusive; both models move by a comparable amount.")
p()

# ---------------------------------------------------------------- E
p("-" * 100)
p("### E. PER-TASK Z DEMAND UNDER R2  (H-03 DA, H-04 lane)")
p("-" * 100)
p()
for task in TASKS:
    ms = [m for m in METRICS if TASK_OF[m] == task]
    p("  %s" % task)
    for m in ms:
        if (m, "R2") in sens:
            d32, d128 = sens[(m, "R2")]
            v = "beyond noise" if abs(d128) > NOISE[m] else "inside noise"
            mono = "monotone-ish" if (d32 > 0 and d128 > d32) else "non-monotone"
            p("    %-10s z16->z32 %+.4f   z16->z128 %+.4f   -> %s, %s" % (
                m, d32, d128, v, mono))
    p()

# ---------------------------------------------------------------- F
p("-" * 100)
p("### F. WHICH EXPLANATIONS ARE STILL OPEN")
p("-" * 100)
p()
p("  R2 removes explanation B (bypass) by construction. The others are NOT")
p("  settled by this table and need the P4A-STEP5 diagnostics:")
p()
for k, txt in [
    ("A", "task genuinely saturated at z=16"),
    ("C", "the encoder, not Z, is the binding constraint"),
    ("D", "the Z -> detection reconstruction is too weak"),
    ("E", "Z carries too little spatial information"),
    ("F", "task gradients conflict and suppress Z utilisation"),
    ("G", "one shared Z is the wrong interface"),
]:
    p("    %s  %s" % (k, txt))
p()

# ---------------------------------------------------------------- paired csv
cols = ["z", "variant", "encoder", "params_M", "flops_G"] + METRICS + ["epochs", "seed", "source"]
with open(PAIRED, "w") as f:
    f.write(",".join(cols) + "\n")
    for z in ZS:
        for tag, src in (("R0", r0), ("R2", r2)):
            r = src.get(z)
            if not r:
                continue
            f.write(",".join([str(z), tag, ENC,
                              r.get("params_M", "NA"), r.get("flops_G", "NA")] +
                             [r.get(m, "NA") for m in METRICS] +
                             [r.get("epochs", "20"), r.get("seed", "0"),
                              r.get("source", "reused")]) + "\n")

with open(ANALYSIS, "w") as f:
    f.write("\n".join(w) + "\n")
print()
print("written:", PAIRED)
print("written:", ANALYSIS)
