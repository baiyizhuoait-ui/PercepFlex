"""Phase 4B-3: apply the preregistered D1-D5 rules to the detection probe.

The rules are transcribed from docs/PHASE5_DET_ANCHOR_PREREGISTRATION.md, which
was written BEFORE any detection cell was trained. This script exists so the
rules are applied mechanically, not argued after seeing the numbers.

Noise floors (external Phase 2-C): mAP50 0.0146, lane_fg 0.0064, da_fg 0.0404.
Baseline is the committed r2u_z16 4-epoch row; it is never rerun.

Usage
-----
    gpu_env/bin/python scripts/phase5_det_decide.py \
        --csv experiments/phase5/phase5_det_probe_results.csv \
        [--sizes experiments/phase5/phase5_det_size_recall.csv] \
        [--out experiments/phase5/phase5_det_probe_decision.txt]
"""
import argparse
import csv
import os

NOISE = {"mAP50": 0.0146, "lane_fg": 0.0064, "da_fg": 0.0404}


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def verdict(gain, noise, what):
    if gain is None:
        return f"{what}: MISSING"
    x = gain / noise
    if x >= 2.0:
        tag = "PASS (>2x noise)"
    elif x >= 1.0:
        tag = "WEAK (1x-2x noise)"
    else:
        tag = "FAIL (<1x noise)"
    return f"{what}: gain {gain:+.4f} = {x:+.2f}x noise -> {tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiments/phase5/phase5_det_probe_results.csv")
    ap.add_argument("--sizes", default="experiments/phase5/phase5_det_size_recall.csv")
    ap.add_argument("--out", default="experiments/phase5/phase5_det_probe_decision.txt")
    ap.add_argument("--baseline-cell", default="r2u_z16")
    args = ap.parse_args()

    rows = load(args.csv)
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append(r)
    base = by_cell.get(args.baseline_cell, [None])[0]
    lines = ["Phase 4B-3 detection probe - preregistered decision",
             f"baseline cell = {args.baseline_cell}", ""]
    if base is None:
        lines.append(f"[FATAL] baseline {args.baseline_cell} not in {args.csv}")
        print("\n".join(lines))
        return
    lines.append(f"baseline: mAP50={base['mAP50']} params={base['params_M']}M "
                 f"flops={base['flops_G']}G lane_fg={base['lane_fg']} da_fg={base['da_fg']}")

    def delta(cell, key):
        """(gain vs baseline, row). (None, None) when the cell is missing or its
        row is present-but-empty (a failed run leaves such a row behind)."""
        for r in by_cell.get(cell, []):
            if f(r, key) is not None and f(base, key) is not None:
                return f(r, key) - f(base, key), r
        return None, None

    # ---- D1 primary (H19) ----
    lines.append("")
    lines.append("D1 (primary, H19): danc mAP50 gain >= 2x noise (0.0292)")
    g1, r1 = delta("danc_z16", "mAP50")
    if g1 is None:
        lines.append("  danc_z16: NOT RUN")
        d1_pass = False
    else:
        lines.append("  " + verdict(g1, NOISE["mAP50"], "danc mAP50"))
        lines.append(f"  mAP50_95: {base['mAP50_95']} -> {r1['mAP50_95']}")
        d1_pass = g1 >= 2 * NOISE["mAP50"]

    # ---- D2 mechanism ----
    lines.append("")
    lines.append("D2 (mechanism): the gain must land on small-object recall.")
    if os.path.exists(args.sizes):
        sz = load(args.sizes)
        pivot = {}
        for r in sz:
            if r["bucket_by"] == "side":
                pivot.setdefault(r["bucket"], {})[r["cell"]] = r
        lines.append(f"  {'bucket':<8}{'n_gt':>8}{'base R':>9}{'danc R':>9}{'dR':>9}"
                     f"{'base AP':>9}{'danc AP':>9}{'dAP':>9}")
        small_dr = None
        for b in ("small", "medium", "large"):
            if b not in pivot:
                continue
            p = pivot[b]
            if "r2u_z16" not in p or "danc_z16" not in p:
                continue
            a, c = p["r2u_z16"], p["danc_z16"]
            dr = float(c["recall@0.5"]) - float(a["recall@0.5"])
            dap = float(c["AP@0.5"]) - float(a["AP@0.5"])
            if b == "small":
                small_dr = dr
            lines.append(f"  {b:<8}{int(a['n_gt']):>8}{float(a['recall@0.5']):>9.4f}"
                         f"{float(c['recall@0.5']):>9.4f}{dr:>+9.4f}"
                         f"{float(a['AP@0.5']):>9.4f}{float(c['AP@0.5']):>9.4f}{dap:>+9.4f}")
        if small_dr is None:
            lines.append("  [INCOMPLETE] small bucket not populated in both cells")
        else:
            if small_dr > 0:
                lines.append(f"  D2 formal: small recall rises ({small_dr:+.4f}) "
                             "-> mechanism CONFIRMED by the rule as written")
            else:
                lines.append(f"  D2 formal: small recall does NOT rise ({small_dr:+.4f}) "
                             "-> mechanism UNCONFIRMED (prereg: reported as such "
                             "regardless of D1)")
            # Where did the gain actually land? The formal rule only asks whether
            # small recall rose; it does not ask whether small is where the gain
            # is. Report the split so the mechanism cannot be over-claimed.
            tot, per = 0.0, {}
            for b in ("small", "medium", "large"):
                if b not in pivot or "r2u_z16" not in pivot[b]:
                    continue
                a, c = pivot[b]["r2u_z16"], pivot[b]["danc_z16"]
                d = (float(c["AP@0.5"]) - float(a["AP@0.5"])) * int(a["n_gt"])
                per[b] = d
                tot += d
            if tot > 0:
                lines.append("  share of the weighted AP gain by bucket "
                             "(sums to 100%):")
                for b in ("small", "medium", "large"):
                    if b in per:
                        lines.append(f"    {b:<8}{100.0 * per[b] / tot:>6.1f}%")
                dom = max(per, key=per.get)
                if dom != "small":
                    lines.append(f"  D2 substantive: the dominant bucket is '{dom}', "
                                 "not 'small' -> the predicted small-object "
                                 "channel is NOT the main mechanism. Report as "
                                 "formally-confirmed / substantively-weak.")
    else:
        lines.append(f"  [PENDING] {args.sizes} not found")

    # ---- D3 sequential (H20) ----
    lines.append("")
    lines.append("D3 (H20): dp2a - danc >= 1x noise (0.0146) -> grid resolution binds")
    gd, rd = delta("dp2a_z16", "mAP50")
    if gd is None or r1 is None:
        lines.append("  dp2a_z16: NOT RUN")
    else:
        lines.append("  " + verdict(gd - g1, NOISE["mAP50"], "dp2a - danc"))
        lines.append("  " + verdict(gd, NOISE["mAP50"], "dp2a vs baseline"))
        lines.append(f"  dp2a params={rd['params_M']}M flops={rd['flops_G']}G "
                     f"(baseline {base['params_M']}M / {base['flops_G']}G)")

    # ---- D4 control ----
    lines.append("")
    lines.append("D4 (control): lane_fg and da_fg move < 2x noise in every cell")
    for cell in ("danc_z16", "dp2a_z16"):
        for key in ("lane_fg", "da_fg"):
            g, r = delta(cell, key)
            if g is None:
                continue
            ok = abs(g) < 2 * NOISE[key]
            lines.append(f"  {cell} {key}: {g:+.4f} = {g/NOISE[key]:+.2f}x noise "
                         f"-> {'PASS' if ok else 'FAIL'}")

    # ---- D5 cost ----
    lines.append("")
    lines.append("D5 (cost): danc params/FLOPs equal to baseline")
    if r1 is not None:
        dp = abs(f(r1, "params_M") - f(base, "params_M"))
        df = abs(f(r1, "flops_G") - f(base, "flops_G"))
        ok = dp < 1e-3 and df < 1e-3
        lines.append(f"  dparams={dp:.4f}M dflops={df:.4f}G -> {'PASS' if ok else 'FAIL'}")

    lines.append("")
    lines.append("READ: a 4-epoch D1 pass authorises a 20-epoch confirmation only;")
    lines.append("      it does not establish the finding (two prior findings in this")
    lines.append("      project reversed between 4 and 20 epochs).")

    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(txt + "\n")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
