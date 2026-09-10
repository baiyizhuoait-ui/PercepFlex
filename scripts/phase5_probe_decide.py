"""P5-STEP7 - apply the pre-registered decision rule to the lane probe.

Why this is a separate script
-----------------------------
The thresholds in docs/PHASE5_LANE_PROBE_PREREGISTRATION.md were fixed before
any number existed. Writing them into code, and running that code, means the
decision is produced by the rule rather than by judgement applied after seeing
the result. This script does not choose which rule to apply; it evaluates all
of them and prints the branch that fires.

It also refuses to over-read: a 4-epoch pass cannot establish H-05b. It can only
authorise the 20-epoch run. The script says so explicitly in its output,
because two previous findings in this project reversed between 4 and 20 epochs.
"""
import csv
import io
import os
import sys

ROOT = "/home/mycode/ai_study/trac"
CSV = os.path.join(ROOT, "experiments", "phase5", "phase5_lane_probe_results.csv")
OUT = os.path.join(ROOT, "experiments", "phase5", "phase5_lane_probe_decision.txt")

# Fixed in the pre-registration. Do not edit to fit a result.
NOISE = {"lane_fg": 0.0064, "mAP50": 0.0146, "da_fg": 0.0404, "lane_mIoU": 0.0117}
BASELINE_CELL = "r2u_z16"
L1_BAR = 0.1829          # baseline lane_fg + 2x noise, quoted in the prereg
BASE_FLOPS = 1.0796      # baseline r2u_z16 GFLOPs

CELLS = ["r2u_z16", "l14up_z16", "l14f1_z16", "lch64_z16"]
# Expected extra GFLOPs from the prereg table, used only if the measured
# flops_G column is missing or non-positive.
FALLBACK_DFLOPS = {"l14up_z16": 0.53, "l14f1_z16": 0.53, "lch64_z16": 0.41}


def load():
    if not os.path.exists(CSV):
        sys.exit("no results csv yet: %s" % CSV)
    rows = list(csv.DictReader(io.open(CSV, encoding="utf-8-sig")))
    return {r["cell"]: r for r in rows}


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main():
    data = load()
    out = []
    say = out.append

    say("Phase 5 lane probe - pre-registered decision")
    say("=" * 72)
    say("Baseline cell: %s (P4A-STEP2 sanity, 4 epochs, not retrained)"
        % BASELINE_CELL)
    say("Noise floor (2x Phase 2-C external): lane_fg %.4f, mAP50 %.4f, "
        "da_fg %.4f" % (2 * NOISE["lane_fg"], 2 * NOISE["mAP50"],
                        2 * NOISE["da_fg"]))
    say("")

    missing = [c for c in CELLS if c not in data]
    if missing:
        say("MISSING CELLS: %s" % ", ".join(missing))
        say("No decision can be issued until they finish.")
        for c in CELLS:
            if c in data:
                r = data[c]
                say("  %-10s lane_fg=%s mAP50=%s da_fg=%s"
                    % (c, r.get("lane_fg"), r.get("mAP50"), r.get("da_fg")))
        write(out)
        return

    base = data[BASELINE_CELL]
    b_lane = f(base, "lane_fg")
    b_map = f(base, "mAP50")
    b_da = f(base, "da_fg")
    b_flops = f(base, "flops_G")

    say("%-10s %8s %8s %8s %8s %9s %9s" % (
        "cell", "lane_fg", "d_lane", "x_noise", "mAP50", "da_fg", "GFLOPs"))
    for c in CELLS:
        r = data[c]
        dl = f(r, "lane_fg") - b_lane
        say("%-10s %8.4f %+8.4f %8.2f %8.4f %8.4f %9.4f" % (
            c, f(r, "lane_fg"), dl, dl / NOISE["lane_fg"],
            f(r, "mAP50"), f(r, "da_fg"), f(r, "flops_G")))
    say("")

    # --- L4: the control --------------------------------------------------
    l4_ok = True
    say("L4 control (detection and DA must not move more than 2x noise):")
    for c in ("l14up_z16", "l14f1_z16", "lch64_z16"):
        r = data[c]
        dmap = f(r, "mAP50") - b_map
        dda = f(r, "da_fg") - b_da
        ok = abs(dmap) < 2 * NOISE["mAP50"] and abs(dda) < 2 * NOISE["da_fg"]
        l4_ok = l4_ok and ok
        say("  %-10s d_mAP50=%+0.4f (%.2fx noise)  d_da_fg=%+0.4f "
            "(%.2fx noise)  %s" % (
                c, dmap, dmap / NOISE["mAP50"], dda, dda / NOISE["da_fg"],
                "ok" if ok else "MOVED"))
    if not l4_ok:
        say("  -> control FAILED. Recorded as task-coupling evidence through Z,")
        say("     not as a broken cell. It weakens any claim that the lane")
        say("     change is isolated.")
    say("")

    # --- L1 / L2 ----------------------------------------------------------
    g14f1 = f(data["l14f1_z16"], "lane_fg") - b_lane
    g14up = f(data["l14up_z16"], "lane_fg") - b_lane
    gch64 = f(data["lch64_z16"], "lane_fg") - b_lane

    l1 = f(data["l14f1_z16"], "lane_fg") >= L1_BAR
    l2 = g14up >= NOISE["lane_fg"]

    say("L1 (H-05b, primary): l14f1 lane_fg >= %.4f -> %s (%.4f, %+.4f)"
        % (L1_BAR, "PASS" if l1 else "FAIL",
           f(data["l14f1_z16"], "lane_fg"), g14f1))
    say("L2 (H-05a):          l14up gain >= 1x noise (%.4f) and < L1 gain -> %s "
        "(%+.4f)" % (NOISE["lane_fg"], "PASS" if (l2 and g14up < g14f1)
                     else "FAIL", g14up))
    say("")

    # --- L3: gain per GFLOP ------------------------------------------------
    def dflops(c):
        d = f(data[c], "flops_G") - b_flops
        if d <= 0:
            d = FALLBACK_DFLOPS.get(c, float("nan"))
        return d

    df1, dfc = dflops("l14f1_z16"), dflops("lch64_z16")
    per1 = g14f1 / df1 if df1 else float("nan")
    perc = gch64 / dfc if dfc else float("nan")
    l3 = per1 > perc
    say("L3 (spatial vs channel, per unit compute):")
    say("  l14f1: gain %+0.4f / %0.4f GFLOPs = %+0.4f per GFLOP" % (
        g14f1, df1, per1))
    say("  lch64: gain %+0.4f / %0.4f GFLOPs = %+0.4f per GFLOP" % (
        gch64, dfc, perc))
    say("  -> %s" % ("PASS: lane is spatially constrained" if l3
                     else "FAIL: resolution buys no more per FLOP than channels"))
    say("")

    # --- the branch --------------------------------------------------------
    say("Decision (rule fixed in the pre-registration)")
    say("-" * 72)
    if l1 and l3:
        say("L1 and L3 -> H-05b SUPPORTED.")
        say("ACTION: extend l14f1_z16 to 20 epochs. Do not run the 1/2 rung.")
    elif l1 and not l3:
        say("L1, not L3 -> resolution helps but buys no more per FLOP than "
            "channels.")
        say("H-05b supported weakly. H-06 is not subordinate. 'Lane is spatial' "
            "cannot be claimed.")
        say("ACTION: hold. Do not extend on this branch yet.")
    elif l2 and not l1:
        say("L2 only -> the gain is upsampling, not information.")
        say("H-05b NOT supported. The 1/4 lateral is not worth its complexity.")
        say("ACTION: do not extend. Reopen H-05a as the operative hypothesis.")
    elif max(g14f1, g14up, gch64) < NOISE["lane_fg"]:
        say("Nothing reaches 1x noise -> H-05b NOT supported at Z=16.")
        say("ACTION: do not extend to 20 epochs on this branch. The next "
            "question is")
        say("whether the 1/4 map itself is too shallow (32 ch, one block) to "
            "carry lane,")
        say("which needs its own registration.")
    else:
        say("Between 1x and 2x noise -> UNRESOLVED.")
        say("ACTION: extend the two most informative cells to 20 epochs. Do "
            "not declare a result.")
    say("")
    say("Standing caveat, restated because it binds this reading:")
    say("two 4-epoch findings in this project reversed by 20 epochs (probe A "
        "detection")
    say("control +0.26x -> -2.53x; reconstruction depth +4.47x -> -0.10x). A "
        "PASS here")
    say("authorises the 20-epoch run. It does not establish H-05b.")
    if not l4_ok:
        say("The L4 control also failed, so even a PASS is conditional on the "
            "coupling")
        say("being understood at 20 epochs.")

    write(out)


def write(out):
    txt = "\n".join(out) + "\n"
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
