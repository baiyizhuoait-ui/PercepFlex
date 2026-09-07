#!/usr/bin/env python
"""Phase 4A Level 2 probe A - evaluate the pre-registered predictions.

Every number is recomputed from experiments/phase4a/phase4B_probeA_results.csv.
The thresholds are NOT chosen here: they are read from the pre-registration
(docs/PHASE4B_PROBEA_PREREGISTRATION.md), which was committed at 35a04e8 before
the runner produced any number. The decision rule is applied as written.
"""
import os, csv, sys

ROOT = "/home/mycode/ai_study/trac"
CSV = os.path.join(ROOT, "experiments/phase4a/phase4B_probeA_results.csv")
OUT = os.path.join(ROOT, "experiments/phase4a/phase4B_probeA_analysis.txt")

# External noise floor, Phase 2-C, 3 seeds @ 4 epochs. Not estimated by this probe.
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}
READ = 2.0  # readability threshold in multiples of the noise floor

rows = list(csv.DictReader(open(CSV)))
C = {r["cell"]: r for r in rows}
assert len(C) == 4, "expected 4 probe cells, got %d" % len(C)

L = []
def p(s=""):
    L.append(s)
    print(s)


def val(cell, m):
    v = C[cell][m]
    return None if v in ("NA", "") else float(v)


def xn(delta, m):
    """delta expressed in multiples of the noise floor."""
    return None if delta is None else delta / NOISE[m]


def fmt(delta, m):
    if delta is None:
        return "NA"
    return "%+.4f (%.2fx noise)" % (delta, xn(delta, m))


p("Phase 4A Level 2 probe A - analysis")
p("=" * 74)
p("4 epochs, seed 0, E-base, 640x640, batch 16. Noise floor external (Phase 2-C,")
p("3 seeds @ 4ep). This probe is single-seed and estimates no variance of its own.")
p("")

p("## Cells")
p("")
p("| cell | z | params | GFLOPs | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg |")
p("|---|---|---|---|---|---|---|---|---|---|")
for c in ("r2u_z16", "r2u_z32", "r3tp_z16", "r3tp_z32"):
    r = C[c]
    p("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        c, r["z"], r["params_M"], r["flops_G"], r["mAP50"], r["mAP50_95"],
        r["da_mIoU"], r["da_fg"], r["lane_mIoU"], r["lane_fg"]))
p("")

# ---------------------------------------------------------------- P1
p("## P1 (z16 arm, lane)  predicted: lane_fg gain >= 2x noise")
d_lfg = val("r3tp_z16", "lane_fg") - val("r2u_z16", "lane_fg")
d_lmi = val("r3tp_z16", "lane_mIoU") - val("r2u_z16", "lane_mIoU")
p("   lane_fg   R3 - R2 = %s   [threshold %+.4f]" % (fmt(d_lfg, "lane_fg"),
                                                     READ * NOISE["lane_fg"]))
p("   lane_mIoU R3 - R2 = %s" % fmt(d_lmi, "lane_mIoU"))
p1 = d_lfg is not None and xn(d_lfg, "lane_fg") >= READ
p("   -> P1 %s" % ("PASS" if p1 else "FAIL"))
p("")

# ---------------------------------------------------------------- P2
p("## P2 (z32 arm, DA)  predicted: squeezing DA 32->16 costs < 2x noise")
d_dfg = val("r3tp_z32", "da_fg") - val("r2u_z32", "da_fg")
d_dmi = val("r3tp_z32", "da_mIoU") - val("r2u_z32", "da_mIoU")
p("   da_fg     R3 - R2 = %s   [threshold |d| < %.4f]" % (fmt(d_dfg, "da_fg"),
                                                          READ * NOISE["da_fg"]))
p("   da_mIoU   R3 - R2 = %s" % fmt(d_dmi, "da_mIoU"))
p2 = abs(xn(d_dfg, "da_fg")) < READ and abs(xn(d_dmi, "da_mIoU")) < READ
p("   -> P2 %s" % ("PASS" if p2 else "FAIL"))
p("   NOTE: both point estimates are NEGATIVE and sit at ~1x noise, so this")
p("   passes the pre-registered 2x criterion while leaving the sign unresolved.")
p("")

# ---------------------------------------------------------------- P3
p("## P3 (cross-arm, PRIMARY)  predicted: R3 z-gap < R2 z-gap - 2x noise")
p("")
p("| metric | R2 gap (z32-z16) | R3 gap (z32-z16) | R3 - R2 | verdict |")
p("|---|---|---|---|---|")
p3 = True
for m in ("mAP50", "mAP50_95", "lane_mIoU", "lane_fg"):
    g2 = val("r2u_z32", m) - val("r2u_z16", m)
    g3 = val("r3tp_z32", m) - val("r3tp_z16", m)
    ok = abs(g3) < abs(g2) - READ * NOISE[m]
    p3 = p3 and ok
    p("| %s | %s | %s | %s | %s |" % (
        m, fmt(g2, m), fmt(g3, m), fmt(g3 - g2, m), "shrunk" if ok else "NOT shrunk"))
p("")
p("   -> P3 %s" % ("PASS" if p3 else "FAIL"))
p("")

# ---------------------------------------------------------------- P4
p("## P4 (detection control)  predicted: det held at 32 -> |R3-R2| < 1x noise")
p4 = True
for z, (a, b) in {"16": ("r3tp_z16", "r2u_z16"), "32": ("r3tp_z32", "r2u_z32")}.items():
    for m in ("mAP50", "mAP50_95"):
        d = val(a, m) - val(b, m)
        ok = abs(xn(d, m)) < 1.0
        p4 = p4 and ok
        p("   z=%s %-9s R3 - R2 = %s  -> %s" % (z, m, fmt(d, m), "ok" if ok else "MOVED"))
p("   -> P4 %s" % ("PASS" if p4 else "FAIL"))
p("")

# ---------------------------------------------------------------- decision
p("## Decision (rule fixed in the pre-registration)")
p("")
if p3:
    dec, why = "A = YES", "P3 holds: the R3 z-gap is smaller than the R2 z-gap."
elif p1 and p2:
    dec, why = "A = YES", "P1 and P2 both hold (secondary route)."
elif p2 and not p1:
    # AMBIGUITY IN THE PRE-REGISTRATION, resolved here rather than hidden.
    # The pre-registration said "A = PARTIAL if P1/P2 hold but P3 fails". Read
    # as "either", that is satisfied, because P2 passed. Read substantively
    # ("per-task width helps"), it is NOT: P2 is a null prediction - it says
    # nothing happens - and P1, the one test that asked for a GAIN, failed.
    # A null is not evidence that the mechanism works.
    dec = "A = PARTIAL (literal 'either' reading) / A = NO (substantive reading)"
    why = ("only P2 holds and P2 is a null, not a gain. P1 (the gain test) and "
           "P3 (the primary test) both fail. No prediction that required "
           "something to IMPROVE was met.")
elif p1 and not p2:
    dec, why = "A = PARTIAL", "P1 holds but P2 and P3 fail."
else:
    dec, why = "A = NO", "neither P3 nor (P1 and P2) holds."
p("   P1=%s  P2=%s  P3=%s  P4=%s" % (p1, p2, p3, p4))
p("   %s" % dec)
p("   -- %s" % why)
p("")
p("   HONEST READING: every prediction that required an IMPROVEMENT failed.")
p("   The single pass (P2) predicted that nothing would happen, and nothing")
p("   measurable did. So the probe produced no positive signal for task-specific")
p("   projection at these widths. Under the user's plan that routes to B,")
p("   except for the power caveat below.")
p("")

# ---------------------------------------------------------------- power
p("## Power diagnosis (how much to trust the null)")
p("")
r2_4 = val("r2u_z16", "mAP50")
p("   r2u_z16 @4ep  mAP50 = %.4f" % r2_4)
p("   r2_z16  @20ep mAP50 = 0.3543 (Phase 4A, same cell)")
p("   4ep sits at %.0f%% of the 20ep level -> the probe is far from converged."
  % (100.0 * r2_4 / 0.3543))
p("")
p("   R2 z-gap mAP50 at 4ep  : %s" % fmt(val("r2u_z32", "mAP50") - val("r2u_z16", "mAP50"), "mAP50"))
p("   R2 z-gap mAP50 at 20ep : +0.0010 (0.14x noise, z16->z32) / +0.0171 (2.34x, z16->z128)")
p("   => the gap P3 was supposed to shrink is itself only ~1x noise at 4ep, so")
p("      P3's failure is weak evidence on its own. P1 is the better-powered test")
p("      because it does not depend on the baseline gap being large.")
p("")

# ---------------------------------------------------------------- efficiency
p("## Efficiency (secondary, not part of the decision rule)")
p("")
for a, b in (("r3tp_z32", "r2u_z32"), ("r3tp_z16", "r2u_z16")):
    fa, fb = float(C[a]["flops_G"]), float(C[b]["flops_G"])
    ma, mb = val(a, "mAP50"), val(b, "mAP50")
    p("   %s vs %s : FLOPs %+.1f%%  mAP50 %s" % (
        a, b, 100.0 * (fa - fb) / fb, fmt(ma - mb, "mAP50")))
p("")

with open(OUT, "w") as fh:
    fh.write("\n".join(L) + "\n")
print("written:", OUT)
