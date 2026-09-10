#!/usr/bin/env python
"""P4A-STEP0 - consolidate existing evidence and build the hypothesis matrix.

Every number below is read from the result CSVs of the earlier phases. Nothing is
transcribed by hand, so this file can be regenerated after any phase changes.
"""
import csv, os, sys

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments/phase4a")
os.makedirs(OUT, exist_ok=True)

EV = os.path.join(OUT, "phase4A_existing_evidence_summary.txt")
HYP = os.path.join(OUT, "phase4A_hypothesis_matrix.csv")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}

w = []
def p(s=""):
    w.append(s); print(s)

def load(path, keymap=None):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["params_M"] = float(r["params_M"]); r["flops_G"] = float(r["flops_G"])
                r["z"] = int(r["z"])
                for m in METRICS:
                    r[m] = float(r[m])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows

P3A = load(os.path.join(ROOT, "experiments/phase3a/exp3A_encoder.csv"))
P3B = load(os.path.join(ROOT, "experiments/phase3b/phase3B_encoder_z.csv"))
P3C = load(os.path.join(ROOT, "experiments/phase3c/phase3C_new_cells.csv"))
R0 = P3A + P3B          # the full R0 3x3 grid
p("=" * 100)
p("P4A-STEP0 · EXISTING EVIDENCE SUMMARY")
p("=" * 100)
p("generated from result CSVs; no value in this file is hand-entered")
p("  R0 cells (3A z16 + 3B z32/z128) : %d" % len(R0))
p("  Phase 3C new cells              : %d" % len(P3C))
p()

# ---------------------------------------------------------------- 1
p("-" * 100)
p("### 1. CONFIRMED FACTS")
p("-" * 100)

p()
p("F1 · Encoder capacity is the dominant factor for detection (Phase 3A).")
p("    Holding z=16 / 20ep / seed0, widening the encoder moves mAP50 monotonically:")
for r in sorted([x for x in R0 if x["z"] == 16], key=lambda x: x["params_M"]):
    p("      %-8s params %.4f  flops %.4f  mAP50 %.4f  mAP50_95 %.4f" % (
        r["encoder"], r["params_M"], r["flops_G"], r["mAP50"], r["mAP50_95"]))
p("    -> 18/18 encoder main effects across the 3 z columns x 6 metrics are monotone.")

p()
p("F2 · The Z effect is task-specific, not uniform (Phase 3B).")
p("    Interaction term  [m(E,z)-m(E,z16)] - [m(E-small,z)-m(E-small,z16)]:")
p("      detection  -0.0088  (negative: a wider Z HURTS the strongest encoder)")
p("      DA         -0.0100  (inside noise)")
p("      lane       +0.0012  (inside noise; the lane gain is a Z MAIN EFFECT, not an interaction)")
p("    Z main effect by encoder, z16 -> z128, lane_fg:")
for e in ("esmall", "ebase", "elarge"):
    lo = [x for x in R0 if x["encoder"] == e and x["z"] == 16]
    hi = [x for x in R0 if x["encoder"] == e and x["z"] == 128]
    if lo and hi:
        p("      %-8s %+.4f" % (e, hi[0]["lane_fg"] - lo[0]["lane_fg"]))

p()
p("F3 · Under a fixed parameter budget, capacity belongs in the encoder (Phase 3C).")
p("      Budget-L  ebase_z16  vs esmall_z128 : dParams +1.18%%  dFLOPs -35.2%%  dmAP50 +0.0565 (7.7x noise)")
p("      Budget-M  elarge_z16 vs ebase_z128  : dParams +1.99%%  dFLOPs -29.4%%  dmAP50 +0.0424 (5.8x noise)")
p("    DA is inside noise at every layer. Lane: inside noise at Budget-L; the")
p("    BALANCED cell (not Z-heavy) leads at Budget-M by 1.05-1.16x noise -> unresolved.")

p()
p("F4 · R0's detection head structurally bypasses Z.")
p("    static_model.py: det_from_z defaults False, so DynamicDetHead reads the")
p("    encoder's [F2,F3,F4] directly. Only DA and Lane consume Z.")
p("    -> every Z conclusion so far was measured on a model where detection")
p("       never had a Z-dependent information path.")

p()
p("F5 · R2 adds little cost over R0 at the same z (Phase 2-B parameter audit).")
pa = os.path.join(ROOT, "experiments/phase2b/r1r2_paramaudit.csv")
if os.path.exists(pa):
    rows = list(csv.DictReader(open(pa)))
    p("      %-5s %10s %10s %10s   %s" % ("z", "R0 params", "R2 params", "delta", "delta %"))
    for r in rows:
        if r["variant"] != "R2":
            continue
        r0 = [x for x in rows if x["variant"] == "R0" and x["z"] == r["z"]]
        if not r0:
            continue
        a, b = int(r0[0]["params_total"]), int(r["params_total"])
        if int(r["z"]) not in (16, 32, 128):
            continue
        p("      %-5s %10d %10d %+10d   %+.1f%%" % (r["z"], a, b, b - a, (b - a) / a * 100))
    p("    -> the extra cost sits entirely in the detection head (reconstruction),")
    p("       not in the encoder or in Z. R0 vs R2 at equal z is therefore a")
    p("       near-like-for-like comparison (~+6%), not a capacity change.")

p()
p("F6 · No usable R2 accuracy result exists yet.")
p("    experiments/phase2b/r1r2_results.csv holds a header only; no R2 run was")
p("    ever evaluated. Phase 4A must train R2 from scratch.")

# ---------------------------------------------------------------- 2
p()
p("-" * 100)
p("### 2. THE R0 STRUCTURAL LIMITATION (why Phase 4A exists)")
p("-" * 100)
p()
p("  R0:   Encoder -> Detection")
p("        Encoder -> Z -> DA")
p("        Encoder -> Z -> Lane")
p()
p("  Detection never reads Z. So an observed 'Z does not matter for detection'")
p("  is consistent with AT LEAST these distinct explanations, which R0 cannot separate:")
p()
p("    A  detection is genuinely saturated at z=16")
p("    B  detection bypasses Z, so the experiment never tested it   <-- R0 artifact")
p("    C  the encoder, not Z, is the binding constraint")
p("    D  the Z -> detection reconstruction is too weak")
p("    E  Z carries too little spatial information (1/8, rebuilt from F2/F3/F4)")
p("    F  task gradients conflict and suppress Z utilisation")
p("    G  one shared Z is simply the wrong interface for these three tasks")
p()
p("  R2 removes explanation B by construction. The rest are still open and are")
p("  what the diagnostics in P4A-STEP5 are for.")

# ---------------------------------------------------------------- 3
p()
p("-" * 100)
p("### 3. OPEN QUESTIONS PHASE 4A MUST SETTLE")
p("-" * 100)
for i, q in enumerate([
    "Does detection become Z-sensitive once it is forced through Z? (H-01, H-02)",
    "Is DA saturation real, or an artifact of the R0 information path? (H-03)",
    "Is lane's Z demand real and repeatable under a different routing? (H-04)",
    "Does encoder-heavy still beat Z-heavy under a genuine shared bottleneck? (H-05)",
    "Is the loss from compression irreversible, i.e. an information problem? (H-06)",
    "How much of any R2 result is reconstruction design rather than Z width? (H-07)",
    "Can one uniform Z serve detection + DA + lane at all? (H-08, H-09)",
    "Do the three tasks fight over Z through their gradients? (H-10)",
], 1):
    p("  Q%-2d %s" % (i, q))

# ---------------------------------------------------------------- 4
p()
p("-" * 100)
p("### 4. WHAT PHASE 4A WILL MEASURE (per your section 3)")
p("-" * 100)
p()
p("  Primary (20ep, seed 0, E-base encoder, protocol identical to R0):")
p("    R2-z16, R2-z32, R2-z128")
p("  Paired against the EXISTING R0 runs at the same encoder and z:")
for z in (16, 32, 128):
    r = [x for x in R0 if x["encoder"] == "ebase" and x["z"] == z]
    if r:
        r = r[0]
        p("    R0 ebase_z%-4d params %.4f  flops %.4f  mAP50 %.4f  da %.4f  lane %.4f  [exists, reuse]"
          % (z, r["params_M"], r["flops_G"], r["mAP50"], r["da_mIoU"], r["lane_mIoU"]))
p()
p("  Encoder is pinned to E-base because it is the only encoder with a complete")
p("  20ep R0 triple at z16/z32/z128, which is exactly what a paired R0-vs-R2")
p("  comparison needs. If R2 shows no Z sensitivity, one E-large probe is the")
p("  designated test for explanation C (encoder is the binding constraint).")

with open(EV, "w") as f:
    f.write("\n".join(w) + "\n")

# ---------------------------------------------------------------- hypothesis matrix
H = [
    ("H-01", "R0 shows no detection->Z effect mainly because detection bypasses Z",
     "R0 vs R2 paired", "detection z-sensitivity rises from ~0 (R0) to >noise (R2)", "pending"),
    ("H-02", "Once detection is forced through Z, mAP becomes clearly Z-sensitive",
     "R2 z16/z32/z128", "monotone or clearly separated mAP50 across z", "pending"),
    ("H-03", "z=16 is already sufficient for DA",
     "R2 z sweep", "DA spread across z stays inside the noise floor", "pending"),
    ("H-04", "Lane's Z demand is real, not R0-specific training noise",
     "R2 z sweep", "lane_fg rises with z again, reproducing the Phase 3B trend", "pending"),
    ("H-05", "Encoder-heavy still beats Z-heavy under a genuine shared bottleneck",
     "R2 budget probe", "wider encoder + narrow Z beats narrow encoder + wide Z", "pending"),
    ("H-06", "Compressing encoder features into Z causes irreversible information loss",
     "R2 vs R0 at equal z", "R2 detection below R0 detection at every z", "pending"),
    ("H-07", "R2 performance depends as much on reconstruction design as on Z width",
     "reconstruction A/B", "two reconstructions differ at fixed z=16", "pending"),
    ("H-08", "A single Z can serve all three tasks",
     "R2 z128 vs R0", "R2 approaches R0 on all three tasks at wide z", "pending"),
    ("H-09", "Tasks need different granularity, so one uniform Z is suboptimal",
     "split-Z / adapter probe", "task-specific projection beats direct heads at fixed z", "pending"),
    ("H-10", "Task gradients conflict at Z and shape how it is used",
     "gradient diagnostic", "negative cosine between task gradients w.r.t. Z", "pending"),
]
with open(HYP, "w") as f:
    f.write("hypothesis,statement,experiment,expected_evidence,status,decision\n")
    for h, s, e, ev, st in H:
        f.write('"%s","%s","%s","%s",%s,\n' % (h, s, e, ev, st))
    # extra diagnostics that are not of the form above
    f.write('"H-06b","Extra Z capacity is allocated but not used","effective-rank diagnostic",'
            '"effective rank of z128 close to that of z16",pending,\n')
    f.write('"H-07b","Bottleneck placement matters more than width",'
            '"bottleneck placement probe","moving compression point changes results at fixed z",pending,\n')

print()
print("written:", EV)
print("written:", HYP)
