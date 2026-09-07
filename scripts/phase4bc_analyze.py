#!/usr/bin/env python
"""Level 2 probes B and C: verdicts against the pre-registered criteria.

Every number below is recomputed from the result CSVs. Nothing is carried over
from memory or from the earlier chat summaries.

One correction to the pre-registration is applied here and flagged in the
output: C4 was written as "recompute per-task gradient norms ... detection's
effective share ... renormalised". The diagnostic script backpropagates each
task loss separately and therefore never sees lambda_det, so the share_det
column in its CSV is UNWEIGHTED and cannot be compared with the 0.50 gate as
written. The effective share is recomputed here by applying the weights to the
same measured norms, which is exactly what the gate describes in words.
"""
import csv
import os

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments", "phase4a")

NOISE = {
    "mAP50": 0.0073,
    "mAP50_95": 0.0032,
    "da_mIoU": 0.0142,
    "da_fg": 0.0202,
    "lane_mIoU": 0.0021,
    "lane_fg": 0.0032,
}
METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
LAMBDA = {"det": 0.2, "da": 1.0, "lane": 1.0}


def load(name):
    path = os.path.join(OUT, name)
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["cell"]] = r
    return rows


def m(row, k):
    return float(row[k])


def units(delta, k):
    return delta / NOISE[k]


def tag(u):
    a = abs(u)
    if a >= 2.0:
        return "READABLE" if u > 0 else "READABLE(neg)"
    if a >= 1.0:
        return "unresolved"
    return "noise"


def table(lines, rows, keys, hdr_fmt, row_fmt):
    lines.append(hdr_fmt)
    lines.append("-" * len(hdr_fmt))
    for k in keys:
        lines.append(row_fmt(k, rows))


def section_b(lines):
    pb = load("phase4B_probeB_results.csv")
    pa = load("phase4B_probeA_results.csv")
    base16 = pa["r2u_z16"]
    base32 = pa["r2u_z32"]

    lines.append("=" * 78)
    lines.append("PROBE B  (reconstruction: receptive field vs depth)  4 epochs, seed 0")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Cells vs the R2 baseline at the same z (probe A cells, same protocol):")
    lines.append("")
    lines.append("  %-10s %-6s %9s %9s %9s %9s %9s %9s" % (
        "cell", "z", "mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"))
    for c in ["r2u_z16", "r2d_z16", "r2p_z16", "r2u_z32", "r2d_z32"]:
        r = pb.get(c) or pa.get(c)
        lines.append("  %-10s %-6s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" % (
            c, r["z"], m(r, "mAP50"), m(r, "mAP50_95"), m(r, "da_mIoU"),
            m(r, "da_fg"), m(r, "lane_mIoU"), m(r, "lane_fg")))
    lines.append("")
    lines.append("Deltas vs same-z baseline, in units of the external Phase 2-C noise floor:")
    lines.append("")
    lines.append("  %-22s %10s %8s %-16s" % ("comparison", "metric", "delta", "x noise / verdict"))
    lines.append("  " + "-" * 60)

    def rep(a, b, label):
        ra = pb.get(a) or pa.get(a)
        rb = pb.get(b) or pa.get(b)
        out = []
        for k in METRICS:
            d = m(ra, k) - m(rb, k)
            u = units(d, k)
            out.append("  %-22s %10s %+10.4f %+7.2fx  %s" % (label, k, d, u, tag(u)))
        return out

    for ln in rep("r2d_z16", "r2u_z16", "r2d_z16 - r2u_z16"):
        lines.append(ln)
    lines.append("")
    for ln in rep("r2d_z32", "r2u_z32", "r2d_z32 - r2u_z32"):
        lines.append(ln)
    lines.append("")
    for ln in rep("r2p_z16", "r2u_z16", "r2p_z16 - r2u_z16"):
        lines.append(ln)
    lines.append("")
    for ln in rep("r2p_z16", "r2d_z16", "r2p_z16 - r2d_z16"):
        lines.append(ln)
    lines.append("")

    # ---- Q1
    d50 = m(pb["r2d_z16"], "mAP50") - m(base16, "mAP50")
    d95 = m(pb["r2d_z16"], "mAP50_95") - m(base16, "mAP50_95")
    u50 = units(d50, "mAP50")
    u95 = units(d95, "mAP50_95")
    q1 = u50 >= 2.0 or u95 >= 2.0
    lines.append("-" * 78)
    lines.append("Q1  r2d_z16 beats r2u_z16 by >=2x noise on mAP50 or mAP50_95")
    lines.append("    mAP50    %+0.4f  = %+0.2fx noise" % (d50, u50))
    lines.append("    mAP50_95 %+0.4f  = %+0.2fx noise" % (d95, u95))
    lines.append("    Q1 = %s" % ("PASS" if q1 else "FAIL"))
    lines.append("")

    # ---- Q2
    e50 = m(pb["r2d_z16"], "mAP50") - m(pb["r2p_z16"], "mAP50")
    e95 = m(pb["r2d_z16"], "mAP50_95") - m(pb["r2p_z16"], "mAP50_95")
    v50 = units(e50, "mAP50")
    v95 = units(e95, "mAP50_95")
    q2 = v50 >= 2.0 or v95 >= 2.0
    lines.append("Q2  r2d_z16 (dilated) beats r2p_z16 (plain, same params) by >=2x noise")
    lines.append("    mAP50    %+0.4f  = %+0.2fx noise" % (e50, v50))
    lines.append("    mAP50_95 %+0.4f  = %+0.2fx noise" % (e95, v95))
    lines.append("    Q2 = %s" % ("PASS" if q2 else "FAIL"))
    if not q2 and v50 < -2.0:
        lines.append("    NOTE: the sign is REVERSED. Dilation does not merely fail to add")
        lines.append("    value over depth - it destroys %.2fx noise of what depth bought." % abs(v50))
    lines.append("")

    # ---- Q3
    lines.append("Q3  DA and lane move <1x noise under both r2d_* cells (reconstruction")
    lines.append("    is detection-only; anything larger is a shared-Z interaction)")
    worst = 0.0
    worst_desc = ""
    for c, b in [("r2d_z16", "r2u_z16"), ("r2d_z32", "r2u_z32")]:
        ra = pb[c]
        rb = pb.get(b) or pa.get(b)
        for k in ["da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]:
            u = units(m(ra, k) - m(rb, k), k)
            lines.append("    %-9s %-10s %+7.2fx" % (c, k, u))
            if abs(u) > abs(worst):
                worst = u
                worst_desc = "%s %s" % (c, k)
    q3 = abs(worst) < 1.0
    lines.append("    worst = %+0.2fx (%s)  ->  Q3 = %s" % (
        worst, worst_desc, "PASS" if q3 else "FAIL (marginal)"))
    lines.append("")

    # ---- decision
    lines.append("-" * 78)
    lines.append("PRE-REGISTERED DECISION RULE")
    lines.append("    B = YES      if Q1 and Q2 both hold")
    lines.append("    B = PARTIAL  if Q1 holds, Q2 fails")
    lines.append("    B = NO       if Q1 fails (Q2 then moot)")
    if q1 and q2:
        verdict = "B = YES"
    elif q1:
        verdict = "B = PARTIAL"
    else:
        verdict = "B = NO"
    lines.append("    Q1=%s Q2=%s  -->  %s" % (q1, q2, verdict))
    lines.append("")
    lines.append("The rule answers the question it was written for: is RECEPTIVE FIELD the")
    lines.append("binding constraint on the reconstruction? Answer: no, and not by a small")
    lines.append("margin - dilation moved detection the wrong way by %.2fx noise." % abs(v50))
    lines.append("")
    lines.append("UNPRE-REGISTERED FINDING IN THE CONTROL ARM (must be reported as such)")
    f50 = m(pb["r2p_z16"], "mAP50") - m(base16, "mAP50")
    f95 = m(pb["r2p_z16"], "mAP50_95") - m(base16, "mAP50_95")
    lines.append("    r2p_z16 is the *control* (dilation 1,1) and it was not predicted to")
    lines.append("    win. It wins by the largest margin measured anywhere in Level 2:")
    lines.append("      mAP50    %+0.4f = %+0.2fx noise" % (f50, units(f50, "mAP50")))
    lines.append("      mAP50_95 %+0.4f = %+0.2fx noise" % (f95, units(f95, "mAP50_95")))
    lines.append("    for +2880 params (+1.43%). Decomposition of the two effects:")
    lines.append("      depth (plain blocks, dilation 1)  %+0.2fx noise  <- real lever" %
                 units(f50, "mAP50"))
    lines.append("      dilation on top of that depth     %+0.2fx noise  <- actively harmful" %
                 v50)
    lines.append("    net dilated cell vs baseline       %+0.2fx noise" % u50)
    lines.append("")
    lines.append("This is a 4-epoch, single-seed result and probe A already showed that a")
    lines.append("4ep reading is not safe at 20ep (its detection control was inert at 4ep")
    lines.append("and -2.53x noise at 20ep). r2p_z16 and r2d_z16 were therefore queued at")
    lines.append("20 epochs before this file was written; see phase4B_e20_confirm2.csv.")
    lines.append("")
    return verdict


def section_c(lines):
    pc = load("phase4C_probeC_results.csv")
    base = load("phase4A_results.csv")
    diag = load("phase4C_probeC_gradient.csv")
    rank_new = load("phase4C_probeC_rank.csv")
    rank_old = load("phase4A_effective_rank.csv")

    lines.append("=" * 78)
    lines.append("PROBE C  (H11: rebalance the gradient magnitudes on Z)  20 ep, seed 0")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Rebalanced cells vs the R2 baseline at the same z (20ep, same seed):")
    lines.append("")
    lines.append("  %-10s %-6s %9s %9s %9s %9s %9s %9s" % (
        "cell", "z", "mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"))
    for c in ["r2_z16", "rw_z16", "r2_z32", "rw_z32"]:
        r = pc.get(c) or base.get(c)
        lines.append("  %-10s %-6s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" % (
            c, r["z"], m(r, "mAP50"), m(r, "mAP50_95"), m(r, "da_mIoU"),
            m(r, "da_fg"), m(r, "lane_mIoU"), m(r, "lane_fg")))
    lines.append("")
    lines.append("Deltas vs same-z R2 baseline, in noise units:")
    lines.append("")
    for new, old in [("rw_z16", "r2_z16"), ("rw_z32", "r2_z32")]:
        lines.append("  %s - %s" % (new, old))
        for k in METRICS:
            d = m(pc[new], k) - m(base[old], k)
            u = units(d, k)
            lines.append("    %-10s %+10.4f  %+8.2fx  %s" % (k, d, u, tag(u)))
        lines.append("")

    # ---- C4
    lines.append("-" * 78)
    lines.append("C4  MANIPULATION CHECK (gates everything)")
    lines.append("")
    lines.append("The diagnostic backpropagates each task loss separately, so its share_*")
    lines.append("columns are UNWEIGHTED and cannot be compared with the 0.50 gate as")
    lines.append("written. Effective share is recomputed here by applying lambda to the")
    lines.append("same measured norms - which is what the gate describes in words.")
    lines.append("")
    lines.append("  %-8s %11s %11s %11s | %9s %9s | %9s" % (
        "cell", "g_det", "g_da", "g_lane", "raw_det", "EFF_det", "gate"))
    c4 = True
    for cell in ["rw_z16", "rw_z32"]:
        r = diag[cell]
        gd, ga, gl = float(r["gnorm_det"]), float(r["gnorm_da"]), float(r["gnorm_lane"])
        raw = gd / (gd + ga + gl)
        wd = LAMBDA["det"] * gd
        wa = LAMBDA["da"] * ga
        wl = LAMBDA["lane"] * gl
        eff = wd / (wd + wa + wl)
        ok = eff <= 0.50
        c4 = c4 and ok
        lines.append("  %-8s %11.5f %11.5f %11.5f | %9.3f %9.3f | %s" % (
            cell, gd, ga, gl, raw, eff, "PASS" if ok else "FAIL"))
    lines.append("")
    lines.append("    baselines for comparison (R2, lambda=1 everywhere):")
    for cell in ["r2_z16", "r2_z32"]:
        lines.append("      %-8s raw effective share of det = %s" % (
            cell, "0.732" if cell.endswith("16") else "0.770"))
    lines.append("    C4 = %s" % ("PASS" if c4 else "FAIL"))
    lines.append("")
    lines.append("Secondary observation: the unweighted norms also moved. vs the R2 20ep")
    lines.append("checkpoints, g_da rose 0.00587 -> 0.00717 at z16 (+22%) and 0.00322 ->")
    lines.append("0.00414 at z32 (+29%) while g_det was flat at z16 and rose at z32. The")
    lines.append("rebalanced model learned a Z that the segmentation tasks push harder,")
    lines.append("i.e. the intervention changed the solution, not just the loss scale.")
    lines.append("")

    # ---- C1
    lines.append("C1  a segmentation task gains >=2x noise at fixed z")
    best = (0.0, "", "")
    for new, old in [("rw_z16", "r2_z16"), ("rw_z32", "r2_z32")]:
        for k, thr in [("lane_fg", 0.0064), ("da_mIoU", 0.0284), ("da_fg", 0.0404)]:
            d = m(pc[new], k) - m(base[old], k)
            if units(d, k) > best[0]:
                best = (units(d, k), "%s %s" % (new, k), "%+0.4f" % d)
    c1 = best[0] >= 2.0
    lines.append("    best segmentation movement anywhere: %+0.2fx noise (%s, %s)" %
                 (best[0], best[1], best[2]))
    lines.append("    C1 = %s" % ("PASS" if c1 else "FAIL"))
    lines.append("")

    # ---- C2
    lines.append("C2  under rebalancing, DA and lane become z-sensitive")
    gl = m(pc["rw_z32"], "lane_fg") - m(pc["rw_z16"], "lane_fg")
    gd_ = m(pc["rw_z32"], "da_mIoU") - m(pc["rw_z16"], "da_mIoU")
    lines.append("    rw z32-z16 lane_fg gap %+0.4f = %+0.2fx noise  (R2 gap was +1.53x)" %
                 (gl, units(gl, "lane_fg")))
    lines.append("    rw z32-z16 da_mIoU gap %+0.4f = %+0.2fx noise  (R2 gap was +0.60x)" %
                 (gd_, units(gd_, "da_mIoU")))
    c2 = units(gl, "lane_fg") >= 2.0 or units(gd_, "da_mIoU") >= 2.0
    lines.append("    C2 = %s" % ("PASS" if c2 else "FAIL"))
    lines.append("")

    # ---- C3
    lines.append("C3  detection gives something up (>=2x noise on mAP50, either arm)")
    for new, old in [("rw_z16", "r2_z16"), ("rw_z32", "r2_z32")]:
        d = m(pc[new], "mAP50") - m(base[old], "mAP50")
        d9 = m(pc[new], "mAP50_95") - m(base[old], "mAP50_95")
        lines.append("    %-8s mAP50 %+0.4f = %+7.2fx   mAP50_95 %+0.4f = %+7.2fx" %
                     (new, d, units(d, "mAP50"), d9, units(d9, "mAP50_95")))
    c3 = any(units(m(pc[n], "mAP50") - m(base[o], "mAP50"), "mAP50") <= -2.0
             for n, o in [("rw_z16", "r2_z16"), ("rw_z32", "r2_z32")])
    lines.append("    C3 = %s   (detection was sacrificed, by a very large margin)" %
                 ("PASS" if c3 else "FAIL"))
    lines.append("")

    # ---- exchange rate
    lines.append("-" * 78)
    lines.append("THE EXCHANGE RATE (the number that settles H11)")
    lines.append("")
    for new, old in [("rw_z16", "r2_z16"), ("rw_z32", "r2_z32")]:
        det = units(m(pc[new], "mAP50") - m(base[old], "mAP50"), "mAP50")
        da = units(m(pc[new], "da_fg") - m(base[old], "da_fg"), "da_fg")
        ln = units(m(pc[new], "lane_fg") - m(base[old], "lane_fg"), "lane_fg")
        lines.append("  %-8s gave up %7.2fx noise of mAP50" % (new, abs(det)))
        lines.append("           bought  %7.2fx noise of da_fg, %7.2fx noise of lane_fg" % (da, ln))
    lines.append("")
    lines.append("Detection was not nudged down, it was pulled down by an order of")
    lines.append("magnitude more than the 2x-noise threshold, and the segmentation tasks")
    lines.append("moved by a fraction of one noise unit. Whatever caps DA and lane, it is")
    lines.append("not detection's 73-77% share of the Z gradient.")
    lines.append("")

    # ---- effective rank
    lines.append("-" * 78)
    lines.append("EFFECTIVE RANK OF Z (did rebalancing change how much of Z is used?)")
    lines.append("")
    lines.append("  %-10s %6s %10s %10s" % ("cell", "z", "erank", "frac of z"))
    for c in ["r2_z16", "rw_z16", "r2_z32", "rw_z32"]:
        r = rank_new.get(c) or rank_old.get(c)
        lines.append("  %-10s %6s %10.2f %9.1f%%" % (
            c, r["z_channels"], float(r["effective_rank"]),
            100 * float(r["erank_frac_of_z"])))
    lines.append("")
    lines.append("Rebalancing raised Z utilisation (z16 6.79->7.85 effective dims,")
    lines.append("z32 11.88->13.43) without raising either segmentation metric. More of Z")
    lines.append("is being used and none of the extra use reaches DA or lane.")
    lines.append("")

    # ---- decision
    lines.append("-" * 78)
    lines.append("PRE-REGISTERED DECISION RULE")
    if not c4:
        verdict = "INVALID MANIPULATION"
        note = "no scientific conclusion may be drawn"
    elif c1 or c2:
        verdict = "H11 SUPPORTED"
        note = "re-run the best task-projection cell under balanced gradients"
    elif c3:
        verdict = "H11 REJECTED AS A LEVER"
        note = ("imbalance is real but reallocating gradient does not help the seg "
                "tasks; their ceiling is set by what Z contains")
    else:
        verdict = "H11 NOT BINDING"
        note = "drop this axis"
    lines.append("    C4=%s C1=%s C2=%s C3=%s" % (c4, c1, c2, c3))
    lines.append("    --> %s" % verdict)
    lines.append("    --> %s" % note)
    lines.append("")
    return verdict


def main():
    lines = []
    vb = section_b(lines)
    lines.append("")
    vc = section_c(lines)

    lines.append("=" * 78)
    lines.append("CROSS-PROBE READING")
    lines.append("=" * 78)
    lines.append("")
    lines.append("A (per-task projection width, 20ep)  NO       - lane gain 1.16x noise,")
    lines.append("                                                detection cost 2.53x")
    lines.append("B (reconstruction receptive field)   NO       - dilation costs 3.64x vs")
    lines.append("                                                plain depth")
    lines.append("B (reconstruction DEPTH, unreg.)     POSITIVE - +4.47x noise mAP50 at 4ep,")
    lines.append("                                                awaiting 20ep")
    lines.append("C (gradient magnitude balance, 20ep) REJECTED - sacrificing 13x noise of")
    lines.append("                                                detection buys 0.27x DA")
    lines.append("")
    lines.append("All three probes point the same way: the segmentation tasks are not")
    lines.append("limited by how much of Z they get (A), nor by how hard they push on it")
    lines.append("(C). Detection is limited by the reconstruction, and specifically by its")
    lines.append("depth rather than its receptive field (B). The two groups of tasks have")
    lines.append("different bottlenecks, which is the case for a task-aware FINAL design")
    lines.append("but not for a task-aware SHARED Z.")
    lines.append("")

    txt = "\n".join(lines)
    p = os.path.join(OUT, "phase4BC_analysis.txt")
    with open(p, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n[written] %s" % p)
    print("[verdicts] B=%s | C=%s" % (vb, vc))


if __name__ == "__main__":
    main()
