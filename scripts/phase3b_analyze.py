#!/usr/bin/env python
"""
Phase 3B · Encoder x Z interaction analysis.

Reads the 3x3 matrix (z=16 column from Phase 3A, z32/z128 columns from
Phase 3B) and answers, in order:

  A. raw matrix
  B. Z main effect        : does widening Z help AT ALL, per encoder?
  C. encoder main effect  : cross-check against Phase 3A
  D. interaction effect   : does the value of Z grow with encoder capacity?
  E. DA/Lane diagnosis    : task saturation, or z=16 bottleneck?
  F. detection diagnosis  : R0 bypasses Z, so det must track the ENCODER
  G. cost analysis        : per-task gain per +0.01M params / +0.1 GFLOP
  H. Pareto frontier      : non-dominated configs, not "highest score wins"
  I. stopping rule        : A / B / C / D

Noise handling: Phase 3B is single-seed (seed=0) and does NOT independently
estimate 20-epoch variance. The Phase 2-C 3-seed noise floor is used only as
an EXTERNAL reference scale, and every verdict states that explicitly.

Outputs:
  experiments/phase3b/phase3B_analysis.txt
  experiments/phase3b/phase3B_interaction.csv
  experiments/phase3b/phase3B_pareto.csv
"""
import csv
import os
import sys

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments", "phase3b")
P3A_CSV = os.path.join(ROOT, "experiments", "phase3a", "exp3A_encoder.csv")
P3B_CSV = os.path.join(OUT, "phase3B_encoder_z.csv")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
TASK_OF = {"mAP50": "detection", "mAP50_95": "detection",
           "da_mIoU": "DA", "da_fg": "DA",
           "lane_mIoU": "lane", "lane_fg": "lane"}

# External reference only -- Phase 2-C, 3 seeds, 4 epochs, pooled stdev.
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}

ENCODERS = ["esmall", "ebase", "elarge"]
ZS = [16, 32, 128]


def load_matrix():
    """Return {(encoder, z): row} for all 9 cells, or as many as exist."""
    cells = {}

    # --- z=16 column: Phase 3A (ebase reused from Phase 2-D expD_z16_e20) ---
    if os.path.exists(P3A_CSV):
        with open(P3A_CSV) as f:
            for r in csv.DictReader(f):
                try:
                    cells[(r["encoder"], 16)] = {
                        "params_M": float(r["params_M"]),
                        "flops_G": float(r["flops_G"]),
                        "peak_mem": r.get("peak_gpu_mem_mib", "NA"),
                        "loss": r.get("final_train_loss", "NA"),
                        "wall": r.get("train_wall_min", "NA"),
                        "source": r.get("source", "trained"),
                        **{m: float(r[m]) for m in METRICS},
                    }
                except (KeyError, ValueError):
                    continue

    # --- z32 / z128 columns: Phase 3B ---
    if os.path.exists(P3B_CSV):
        with open(P3B_CSV) as f:
            for r in csv.DictReader(f):
                if r.get("source") == "failed-no-metrics":
                    continue
                try:
                    cells[(r["encoder"], int(r["z"]))] = {
                        "params_M": float(r["params_M"]),
                        "flops_G": float(r["flops_G"]),
                        "peak_mem": r.get("peak_gpu_mem_mib", "NA"),
                        "loss": r.get("final_train_loss", "NA"),
                        "wall": r.get("train_wall_min", "NA"),
                        "source": r.get("source", "trained"),
                        **{m: float(r[m]) for m in METRICS},
                    }
                except (KeyError, ValueError):
                    continue
    return cells


def fmt(v, nd=4):
    return f"{v:+.{nd}f}" if isinstance(v, float) else str(v)


def main():
    cells = load_matrix()
    have = [(e, z) for e in ENCODERS for z in ZS if (e, z) in cells]
    missing = [(e, z) for e in ENCODERS for z in ZS if (e, z) not in cells]

    L = []
    w = L.append
    w("=" * 100)
    w("PHASE 3B · ENCODER x Z INTERACTION ANALYSIS")
    w("=" * 100)
    w("")
    w(f"cells present: {len(have)}/9")
    if missing:
        w(f"cells MISSING: {', '.join(f'{e}_z{z}' for e, z in missing)}")
        w("  -> analysis below is computed on whatever exists; conclusions are")
        w("     provisional until the matrix is complete.")
    w("")
    w("NOISE HANDLING: Phase 3B is single-seed (seed=0) and does NOT")
    w("independently estimate 20-epoch multi-seed variance. The noise floor")
    w("used below is an EXTERNAL reference from Phase 2-C (3 seeds, 4 epochs,")
    w("pooled stdev). Differences near it are reported but not claimed.")
    w("")

    if len(have) < 4:
        w("NOT ENOUGH CELLS TO ANALYSE.")
        print("\n".join(L))
        return 1

    # ------------------------------------------------------------- A. raw
    w("-" * 100)
    w("### A. RAW MATRIX")
    w("-" * 100)
    hdr = (f"{'cell':<14}{'params_M':>9}{'flops_G':>9}{'mAP50':>8}{'mAP50-95':>9}"
           f"{'da_mIoU':>9}{'da_fg':>8}{'lane_mIoU':>10}{'lane_fg':>9}{'loss':>8}{'src':>10}")
    w(hdr)
    w("-" * 100)
    for e in ENCODERS:
        for z in ZS:
            k = (e, z)
            if k not in cells:
                w(f"{e}_z{z:<9}{'-- missing --'}")
                continue
            c = cells[k]
            w(f"{e}_z{z:<9}{c['params_M']:>9.4f}{c['flops_G']:>9.4f}"
              f"{c['mAP50']:>8.4f}{c['mAP50_95']:>9.4f}{c['da_mIoU']:>9.4f}"
              f"{c['da_fg']:>8.4f}{c['lane_mIoU']:>10.4f}{c['lane_fg']:>9.4f}"
              f"{c['loss']:>8}{c['source'][:9]:>10}")

    # ------------------------------------------------------- B. Z main effect
    w("")
    w("-" * 100)
    w("### B. Z MAIN EFFECT   (delta vs z=16, within each encoder)")
    w("-" * 100)
    w("Is widening Z positive, neutral, or negative? Judge on the whole set of")
    w("metrics, never on a single one.")
    w("")
    z_rows = []
    for e in ENCODERS:
        if (e, 16) not in cells:
            continue
        base = cells[(e, 16)]
        for z in (32, 128):
            if (e, z) not in cells:
                continue
            c = cells[(e, z)]
            for m in METRICS:
                d = c[m] - base[m]
                z_rows.append(dict(encoder=e, z=z, metric=m, task=TASK_OF[m],
                                   base=base[m], val=c[m], delta=d,
                                   noise=NOISE[m],
                                   dp=cells[(e, z)]["params_M"] - base["params_M"],
                                   df=cells[(e, z)]["flops_G"] - base["flops_G"]))

    if z_rows:
        w(f"{'encoder':<9}{'z':>5} {'metric':<10}{'z16':>9}{'value':>9}{'delta':>10}"
          f"{'noise*':>9}{'verdict':>14}")
        w("-" * 100)
        for e in ENCODERS:
            for z in (32, 128):
                rs = [r for r in z_rows if r["encoder"] == e and r["z"] == z]
                if not rs:
                    continue
                for r in rs:
                    v = "above noise" if abs(r["delta"]) > r["noise"] else "within noise"
                    w(f"{e:<9}{z:>5} {r['metric']:<10}{r['base']:>9.4f}{r['val']:>9.4f}"
                      f"{fmt(r['delta']):>10}{r['noise']:>9.4f}{v:>14}")
                w("")
        w("* noise column = Phase 2-C external reference, not a Phase 3B estimate")

        # sign summary
        w("")
        w("Sign summary of Z effect (how many of the 6 metrics move each way):")
        for e in ENCODERS:
            for z in (32, 128):
                rs = [r for r in z_rows if r["encoder"] == e and r["z"] == z]
                if not rs:
                    continue
                pos = sum(1 for r in rs if r["delta"] > 0)
                neg = sum(1 for r in rs if r["delta"] < 0)
                abv = sum(1 for r in rs if abs(r["delta"]) > r["noise"])
                w(f"  {e:<9} z{z:<5} positive {pos}/6   negative {neg}/6   "
                  f"beyond external noise {abv}/6")

    # -------------------------------------------------- C. encoder main effect
    w("")
    w("-" * 100)
    w("### C. ENCODER MAIN EFFECT   (cross-check of Phase 3A, now per Z column)")
    w("-" * 100)
    for z in ZS:
        ks = [(e, z) for e in ENCODERS if (e, z) in cells]
        if len(ks) < 2:
            continue
        w("")
        w(f"  z = {z}")
        w(f"    {'metric':<10}{'E-small':>10}{'E-base':>10}{'E-large':>10}"
          f"{'d(base-sm)':>12}{'d(lg-base)':>12}{'d(lg-sm)':>12}  monotonic")
        for m in METRICS:
            vals = [cells[k][m] for k in ks]
            mono = all(b > a for a, b in zip(vals, vals[1:])) if len(vals) == 3 else None
            if len(vals) == 3:
                sm, bs, lg = vals
                w(f"    {m:<10}{sm:>10.4f}{bs:>10.4f}{lg:>10.4f}"
                  f"{fmt(bs - sm):>12}{fmt(lg - bs):>12}{fmt(lg - sm):>12}"
                  f"  {mono}")
            else:
                w(f"    {m:<10}" + "".join(f"{v:>10.4f}" for v in vals) + "   (incomplete)")

    # ------------------------------------------------------- D. interaction
    w("")
    w("-" * 100)
    w("### D. INTERACTION EFFECT   (the core question)")
    w("-" * 100)
    w("interaction = [m(E,z) - m(E,z16)] - [m(E-small,z) - m(E-small,z16)]")
    w("")
    w("  > 0  -> a stronger encoder makes Z MORE valuable (z=16 is throttling)")
    w("  ~ 0  -> the two factors are independent")
    w("  < 0  -> Z matters LESS as the encoder grows")
    w("")
    inter_rows = []
    for ref in ("ebase", "elarge"):
        for z in (32, 128):
            for m in METRICS:
                need = [(ref, 16), (ref, z), ("esmall", 16), ("esmall", z)]
                if not all(k in cells for k in need):
                    continue
                d_ref = cells[(ref, z)][m] - cells[(ref, 16)][m]
                d_sml = cells[("esmall", z)][m] - cells[("esmall", 16)][m]
                inter_rows.append(dict(encoder=ref, z=z, metric=m, task=TASK_OF[m],
                                       d_ref=d_ref, d_esmall=d_sml,
                                       interaction=d_ref - d_sml, noise=NOISE[m]))

    if inter_rows:
        w(f"{'vs E-small':<11}{'z':>5} {'metric':<10}{'dZ(ref)':>10}{'dZ(small)':>11}"
          f"{'interaction':>13}{'noise*':>9}{'read':>16}")
        w("-" * 100)
        for ref in ("ebase", "elarge"):
            for z in (32, 128):
                rs = [r for r in inter_rows if r["encoder"] == ref and r["z"] == z]
                if not rs:
                    continue
                for r in rs:
                    if abs(r["interaction"]) < r["noise"]:
                        read = "~independent"
                    elif r["interaction"] > 0:
                        read = "Z MORE valuable"
                    else:
                        read = "Z LESS valuable"
                    w(f"{ref:<11}{z:>5} {r['metric']:<10}{fmt(r['d_ref']):>10}"
                      f"{fmt(r['d_esmall']):>11}{fmt(r['interaction']):>13}"
                      f"{r['noise']:>9.4f}{read:>16}")
                w("")
        w("* interaction is compared against the SAME external noise floor; a")
        w("  difference of two single-seed cells is itself noisy, so treat the")
        w("  sign pattern across metrics as the evidence, not any single row.")

        w("")
        w("Interaction by task (mean over metrics in the task, z=128 only):")
        for ref in ("ebase", "elarge"):
            for t in ("detection", "DA", "lane"):
                rs = [r for r in inter_rows if r["encoder"] == ref
                      and r["z"] == 128 and r["task"] == t]
                if rs:
                    mean = sum(r["interaction"] for r in rs) / len(rs)
                    w(f"  {ref:<9} {t:<10} mean interaction = {mean:+.4f}")

    # ------------------------------------------- E. DA/Lane bottleneck
    w("")
    w("-" * 100)
    w("### E. DA / LANE BOTTLENECK DIAGNOSIS")
    w("-" * 100)
    w("Phase 3A saw DA/Lane flatten from E-base to E-large at z=16. Two rival")
    w("explanations: (A) the tasks are genuinely saturated, or (B) z=16 is")
    w("throttling the richer encoder features before they reach the seg heads.")
    w("The discriminator is whether DA/Lane REVIVE when Z widens at E-large.")
    w("")
    if all(("elarge", z) in cells for z in ZS):
        w(f"  {'metric':<10}{'z16':>10}{'z32':>10}{'z128':>10}"
          f"{'d(32-16)':>11}{'d(128-16)':>11}  verdict")
        w("  " + "-" * 66)
        for m in ("da_mIoU", "da_fg", "lane_mIoU", "lane_fg"):
            v16 = cells[("elarge", 16)][m]
            v32 = cells[("elarge", 32)][m]
            v128 = cells[("elarge", 128)][m]
            d32, d128 = v32 - v16, v128 - v16
            revived = abs(d128) > NOISE[m] or abs(d32) > NOISE[m]
            verdict = "REVIVES -> z bottleneck" if revived else "flat -> task saturation"
            w(f"  {m:<10}{v16:>10.4f}{v32:>10.4f}{v128:>10.4f}"
              f"{fmt(d32):>11}{fmt(d128):>11}  {verdict}")
        w("")
        w("  NOTE: even the 'task saturation' branch is only EVIDENCE CONSISTENT")
        w("  WITH saturation at z<=128 / 20 epochs. It is not a proof, and it")
        w("  does not rule out saturation at some larger Z we did not test.")
    else:
        w("  E-large row incomplete -- cannot diagnose yet.")

    # ------------------------------------------------- F. detection diagnosis
    w("")
    w("-" * 100)
    w("### F. DETECTION DIAGNOSIS   (R0 bypasses Z)")
    w("-" * 100)
    w("In R0 the detection head reads encoder F2/F3/F4 directly; it never sees")
    w("Z. So ANY mAP response to z cannot be a direct Z effect -- it can only")
    w("be multi-task gradient coupling / shared training dynamics / indirect")
    w("effects through the shared encoder. Detection's primary driver must be")
    w("the ENCODER. Verify both claims below.")
    w("")
    for m in ("mAP50", "mAP50_95"):
        if all((e, 16) in cells for e in ENCODERS):
            sm, bs, lg = (cells[(e, 16)][m] for e in ENCODERS)
            w(f"  {m}: encoder main effect (z=16)  "
              f"small {sm:.4f} -> base {bs:.4f} -> large {lg:.4f}   "
              f"total {lg - sm:+.4f}")
        if all(("elarge", z) in cells for z in ZS):
            v = [cells[("elarge", z)][m] for z in ZS]
            spread = max(v) - min(v)
            w(f"  {m}: z effect at E-large            "
              f"z16 {v[0]:.4f} / z32 {v[1]:.4f} / z128 {v[2]:.4f}   "
              f"spread {spread:.4f}  (noise {NOISE[m]:.4f})")
            w(f"          -> {'within noise, consistent with bypass' if spread <= NOISE[m] else 'exceeds noise: indirect/coupling effect, NOT direct Z consumption'}")
    w("")
    w("  Do NOT write 'detection consumes the compact representation' anywhere")
    w("  in the R0 report. That statement only becomes available under R2.")

    # ------------------------------------------------------------ G. cost
    w("")
    w("-" * 100)
    w("### G. COMPUTE / PARAMETER TRADE-OFF")
    w("-" * 100)
    ref = ("ebase", 16)
    if ref in cells:
        rb = cells[ref]
        w(f"  reference = E-base + z16  ({rb['params_M']:.4f}M, {rb['flops_G']:.4f}G)")
        w("")
        w(f"  {'cell':<14}{'dParams_M':>10}{'dFLOPs_G':>10}"
          f"{'d mAP50':>10}{'d da_mIoU':>11}{'d lane_mIoU':>13}"
          f"{'per 0.01M':>11}{'per 0.1GF':>10}")
        w("  " + "-" * 76)
        for e in ENCODERS:
            for z in ZS:
                k = (e, z)
                if k not in cells or k == ref:
                    continue
                c = cells[k]
                dp = c["params_M"] - rb["params_M"]
                df = c["flops_G"] - rb["flops_G"]
                dm = c["mAP50"] - rb["mAP50"]
                per_p = dm / (dp * 100) if abs(dp) > 1e-9 else float("nan")
                per_f = dm / (df * 10) if abs(df) > 1e-9 else float("nan")
                w(f"  {e}_z{z:<9}{dp:>+10.4f}{df:>+10.4f}"
                  f"{fmt(dm):>10}{fmt(c['da_mIoU'] - rb['da_mIoU']):>11}"
                  f"{fmt(c['lane_mIoU'] - rb['lane_mIoU']):>13}"
                  f"{per_p:>+11.5f}{per_f:>+10.5f}")
        w("")
        w("  ('per 0.01M' and 'per 0.1GF' are mAP50 only -- do not compare")
        w("   across tasks, whose metrics live on different scales.)")

        w("")
        w("  Per-task unit-cost of the two ways to spend ~0.1M params:")
        for e in ENCODERS:
            if (e, 16) in cells and (e, 128) in cells:
                dp = cells[(e, 128)]["params_M"] - cells[(e, 16)]["params_M"]
                df = cells[(e, 128)]["flops_G"] - cells[(e, 16)]["flops_G"]
                w(f"    {e}: spend on Z (z16->z128)  dParams {dp:+.4f}M  "
                  f"dFLOPs {df:+.4f}G")
                for t in ("detection", "DA", "lane"):
                    ms = [m for m in METRICS if TASK_OF[m] == t]
                    tot = sum(cells[(e, 128)][m] - cells[(e, 16)][m] for m in ms) / len(ms)
                    w(f"        {t:<10} mean d {tot:+.4f}   "
                      f"per 0.01M {tot / (dp * 100) if abs(dp) > 1e-9 else float('nan'):+.5f}")
        if all((e, 16) in cells for e in ENCODERS):
            dp = cells[("elarge", 16)]["params_M"] - cells[("esmall", 16)]["params_M"]
            df = cells[("elarge", 16)]["flops_G"] - cells[("esmall", 16)]["flops_G"]
            w(f"    encoder: spend on encoder (E-small->E-large, z16)  "
              f"dParams {dp:+.4f}M  dFLOPs {df:+.4f}G")
            for t in ("detection", "DA", "lane"):
                ms = [m for m in METRICS if TASK_OF[m] == t]
                tot = sum(cells[("elarge", 16)][m] - cells[("esmall", 16)][m] for m in ms) / len(ms)
                w(f"        {t:<10} mean d {tot:+.4f}   "
                  f"per 0.01M {tot / (dp * 100) if abs(dp) > 1e-9 else float('nan'):+.5f}")

    # ---------------------------------------------------------- H. pareto
    w("")
    w("-" * 100)
    w("### H. PARETO FRONTIER")
    w("-" * 100)
    w("The highest-scoring cell is NOT automatically the best. A cell is")
    w("non-dominated only if no other cell costs less (params AND FLOPs) while")
    w("scoring at least as well on ALL SIX metrics.")
    w("")
    present = [(e, z) for e in ENCODERS for z in ZS if (e, z) in cells]
    TOL = 1e-4  # ignore differences at the 4th decimal (below every noise floor)

    def dominates(a, b):
        ca, cb = cells[a], cells[b]
        if ca["params_M"] > cb["params_M"] + TOL:
            return False
        if ca["flops_G"] > cb["flops_G"] + TOL:
            return False
        return all(ca[m] >= cb[m] - TOL for m in METRICS)

    frontier = [k for k in present if not any(dominates(o, k) for o in present if o != k)]
    dominated = [k for k in present if k not in frontier]

    w("  NON-DOMINATED (Pareto frontier):")
    for k in frontier:
        c = cells[k]
        w(f"    {k[0]}_z{k[1]:<8} params {c['params_M']:.4f}M  flops {c['flops_G']:.4f}G  "
          f"mAP50 {c['mAP50']:.4f}  da {c['da_mIoU']:.4f}  lane {c['lane_mIoU']:.4f}")
    w("")
    w("  DOMINATED (some cell is cheaper and at least as accurate):")
    for k in dominated:
        by = [f"{o[0]}_z{o[1]}" for o in present if o != k and dominates(o, k)]
        c = cells[k]
        w(f"    {k[0]}_z{k[1]:<8} params {c['params_M']:.4f}M  flops {c['flops_G']:.4f}G  "
          f"dominated by {', '.join(by)}")

    with open(os.path.join(OUT, "phase3B_pareto.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["encoder", "z", "params_M", "flops_G", *METRICS,
                     "pareto_status", "dominated_by"])
        for k in present:
            c = cells[k]
            by = ";".join(f"{o[0]}_z{o[1]}" for o in present if o != k and dominates(o, k))
            wr.writerow([k[0], k[1], f"{c['params_M']:.4f}", f"{c['flops_G']:.4f}",
                         *[f"{c[m]:.4f}" for m in METRICS],
                         "frontier" if k in frontier else "dominated", by or ""])

    # --------------------------------------------------- I. stopping rule
    w("")
    w("-" * 100)
    w("### I. STOPPING RULE  (A / B / C / D)")
    w("-" * 100)
    if missing:
        w("  Matrix incomplete -- cannot apply the stopping rule yet.")
    else:
        da_lane = ["da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
        # Does widening Z revive DA/Lane, and more so at E-large than E-small?
        def dz(e, m, z=128):
            return cells[(e, z)][m] - cells[(e, 16)][m]

        rev_lg = sum(abs(dz("elarge", m)) > NOISE[m] for m in da_lane)
        rev_sm = sum(abs(dz("esmall", m)) > NOISE[m] for m in da_lane)
        z_any = sum(abs(dz(e, m)) > NOISE[m]
                    for e in ENCODERS for m in METRICS)
        signs = [1 if dz(e, m) > 0 else -1
                 for e in ENCODERS for m in METRICS if (e, 128) in cells]

        w(f"  DA/Lane metrics revived by z128 at E-large : {rev_lg}/4")
        w(f"  DA/Lane metrics revived by z128 at E-small : {rev_sm}/4")
        w(f"  any metric moved by z128 anywhere          : {z_any}/18")
        w("")
        if rev_lg >= 2 and rev_lg > rev_sm:
            rule = "A"
            w("  ==> RULE A: E-large + wider Z revives DA/Lane more than E-small.")
            w("      There is an Encoder x Z interaction; z=16 was throttling.")
            w("      NEXT: Phase 3C fixed-budget allocation")
            w("      (encoder-heavy / balanced / Z-heavy).")
        elif rev_lg == 0 and rev_sm == 0 and z_any <= 2:
            rule = "B"
            w("  ==> RULE B: DA/Lane stay flat across z at E-large, and Z does")
            w("      essentially nothing anywhere. Detection tracks the encoder.")
            w("      Supports 'Z saturates fast; the encoder is the main'")
            w("      bottleneck'. NEXT: go straight to fixed-budget allocation;")
            w("      do NOT widen the Z search space.")
        elif z_any >= 6:
            rule = "C"
            w("  ==> RULE C: Z produces real gains in several places, so Phase 2-D's")
            w("      'Z is saturated' must be RE-SCOPED to 'saturated under the")
            w("      baseline encoder'. Stop saying z=16 is universally")
            w("      sufficient. NEXT: fixed-budget comparison first -- do not")
            w("      simply keep growing z.")
        else:
            rule = "D"
            w("  ==> RULE D: the z effect is inconsistent -- direction flips")
            w("      between encoders/metrics. NO structural conclusion.")
            w("      Audit: implementation, training stability, checkpoints,")
            w("      metric pipeline, loss, dataset, config consistency.")
            w("      Do not select runs and then declare a conclusion.")
        w("")
        w(f"  (rule chosen: {rule})")

    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "phase3B_analysis.txt"), "w") as f:
        f.write(txt)
    print(txt)

    # ------------------------------------------------- interaction CSV
    if inter_rows:
        with open(os.path.join(OUT, "phase3B_interaction.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["encoder", "z", "task", "metric",
                         "delta_Z_vs_z16", "delta_Z_esmall", "interaction",
                         "noise_ref_phase2c", "beyond_noise"])
            for r in inter_rows:
                wr.writerow([r["encoder"], r["z"], r["task"], r["metric"],
                             f"{r['d_ref']:+.4f}", f"{r['d_esmall']:+.4f}",
                             f"{r['interaction']:+.4f}", f"{r['noise']:.4f}",
                             "yes" if abs(r["interaction"]) > r["noise"] else "no"])
        print(f"[ok] wrote {OUT}/phase3B_interaction.csv")
    print(f"[ok] wrote {OUT}/phase3B_analysis.txt")
    print(f"[ok] wrote {OUT}/phase3B_pareto.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
