#!/usr/bin/env python3
"""Phase 3A · Encoder-capacity analysis.

Answers, from data only:
  1. Does performance vary monotonically (or near-monotonically) with encoder
     capacity?
  2. Is the encoder-capacity gain clearly larger than the Phase 2-D Z spread?
  3. Does this preliminarily support "spend parameters on the encoder rather
     than on Z"?
  4. Encoder-sensitivity verdict, including saturation patterns.

Also reports unit-cost efficiency (gain per +0.01M params, per +0.1 GFLOP)
because absolute best score is not the point -- allocation efficiency is.

Reference baselines (both at 20ep, seed=0):
  - Phase 2-D Z sweep @ fixed baseline encoder, z = 16/32/128
  - 3-seed noise floor from Phase 2-C Exp A (conservative proxy)
"""
import csv, io, os
from statistics import mean

ROOT = "/home/mycode/ai_study/trac"
OUTD = os.path.join(ROOT, "experiments", "phase3a")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
ORDER = ["esmall", "ebase", "elarge"]
LABEL = {"esmall": "E-small", "ebase": "E-base", "elarge": "E-large"}

# Phase 2-D @20ep, seed=0, baseline encoder, varying z (the Z-effect reference)
Z20 = {
    16:  {"mAP50": 0.3204, "mAP50_95": 0.1149, "da_mIoU": 0.8564,
          "da_fg": 0.7723, "lane_mIoU": 0.5867, "lane_fg": 0.1973},
    32:  {"mAP50": 0.3197, "mAP50_95": 0.1147, "da_mIoU": 0.8545,
          "da_fg": 0.7696, "lane_mIoU": 0.5863, "lane_fg": 0.1973},
    128: {"mAP50": 0.3224, "mAP50_95": 0.1157, "da_mIoU": 0.8553,
          "da_fg": 0.7708, "lane_mIoU": 0.5879, "lane_fg": 0.2005},
}
# Conservative noise floor (3-seed @4ep, pooled stdev)
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}


def rd(p):
    return list(csv.DictReader(io.open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    rows = rd(os.path.join(OUTD, "exp3A_encoder.csv"))
    if not rows:
        print("no data yet"); return
    cell = {}
    for r in rows:
        try:
            cell[r["encoder"]] = {
                **{m: float(r[m]) for m in METRICS},
                "params_M": float(r["params_M"]), "flops_G": float(r["flops_G"]),
                "stem": r["stem"], "stages": r["stages"],
                "mem": r["peak_gpu_mem_mib"], "loss": r["final_train_loss"],
                "wall": r["train_wall_min"], "source": r["source"],
            }
        except (ValueError, KeyError):
            pass

    have = [c for c in ORDER if c in cell]
    L = []
    P = L.append
    P("=" * 96)
    P("PHASE 3A · ENCODER CAPACITY ANALYSIS")
    P("=" * 96)
    if len(have) < len(ORDER):
        P(f"\n[PARTIAL] cells present: {have}  (missing: "
          f"{[c for c in ORDER if c not in have]})")
        P("Conclusions below are provisional until all three cells land.\n")

    # ---------- main table ----------
    P("\n### Phase 3A TABLE  (z=16, 20ep, seed=0; only encoder width varies)\n")
    hdr = (f"{'Encoder':<9}{'stem':>5}{'stages':>20}{'Params':>9}{'FLOPs':>8}"
           f"{'mAP50':>9}{'mAP50-95':>10}{'DA mIoU':>9}{'DA fg':>9}"
           f"{'Lane mIoU':>11}{'Lane fg':>9}")
    P(hdr)
    P("-" * 96)
    for c in ORDER:
        if c not in cell:
            P(f"{LABEL[c]:<9}{'—':>5}{'—':>20}{'—':>9}{'—':>8}{'—':>9}{'—':>10}"
              f"{'—':>9}{'—':>9}{'—':>11}{'—':>9}")
            continue
        r = cell[c]
        P(f"{LABEL[c]:<9}{r['stem']:>5}{r['stages']:>20}{r['params_M']:>9.4f}"
          f"{r['flops_G']:>8.3f}{r['mAP50']:>9.4f}{r['mAP50_95']:>10.4f}"
          f"{r['da_mIoU']:>9.4f}{r['da_fg']:>9.4f}{r['lane_mIoU']:>11.4f}"
          f"{r['lane_fg']:>9.4f}")

    # ---------- cost / runtime ----------
    P("\n### COST & RUNTIME\n")
    P(f"{'Encoder':<9}{'Params M':>10}{'FLOPs G':>10}{'peak mem MiB':>14}"
      f"{'final loss':>12}{'train min':>11}{'source':>34}")
    P("-" * 96)
    for c in ORDER:
        if c not in cell:
            continue
        r = cell[c]
        P(f"{LABEL[c]:<9}{r['params_M']:>10.4f}{r['flops_G']:>10.3f}"
          f"{r['mem']:>14}{r['loss']:>12}{r['wall']:>11}{r['source']:>34}")

    if len(have) < 2:
        io.open(os.path.join(OUTD, "exp3A_analysis.txt"), "w",
                encoding="utf-8").write("\n".join(L) + "\n")
        print("\n".join(L)); return

    # ---------- Q1 monotonicity ----------
    P("\n### Q1  MONOTONICITY with encoder capacity\n")
    P(f"{'metric':<11}{'E-small':>10}{'E-base':>10}{'E-large':>10}"
      f"{'Δ(base-sm)':>12}{'Δ(lg-base)':>12}{'Δ(lg-sm)':>11}{'monotonic':>11}")
    P("-" * 96)
    mono_flags = {}
    for m in METRICS:
        vs = [cell[c][m] for c in ORDER if c in cell]
        if len(vs) < 2:
            continue
        d1 = cell["ebase"][m] - cell["esmall"][m] if {"ebase", "esmall"} <= set(have) else None
        d2 = cell["elarge"][m] - cell["ebase"][m] if {"elarge", "ebase"} <= set(have) else None
        d3 = cell["elarge"][m] - cell["esmall"][m] if {"elarge", "esmall"} <= set(have) else None
        if len(vs) == 3:
            mono = (cell["esmall"][m] <= cell["ebase"][m] <= cell["elarge"][m]) or \
                   (cell["esmall"][m] >= cell["ebase"][m] >= cell["elarge"][m])
            mono_flags[m] = mono
        else:
            mono = None
        f = lambda v: f"{v:+.4f}" if v is not None else "  n/a"
        P(f"{m:<11}{cell['esmall'][m]:>10.4f}{cell['ebase'][m]:>10.4f}"
          f"{cell['elarge'][m]:>10.4f}{f(d1):>12}{f(d2):>12}{f(d3):>11}"
          f"{str(mono):>11}" if len(vs) == 3 else
          f"{m:<11}" + "".join(f"{v:>10.4f}" for v in vs))

    # ---------- Q2 encoder effect vs Z effect ----------
    P("\n### Q2  ΔEncoder vs ΔZ   (both at 20ep, seed=0)\n")
    zs = sorted(Z20)
    P(f"{'metric':<11}{'ΔZ spread':>11}{'Z noise':>10}"
      f"{'ΔEnc(sm→lg)':>13}{'ΔEnc/ΔZ':>10}{'enc vs noise':>22}")
    P("-" * 96)
    q2 = {}
    for m in METRICS:
        zspread = max(Z20[z][m] for z in zs) - min(Z20[z][m] for z in zs)
        if {"esmall", "elarge"} <= set(have):
            denc = cell["elarge"][m] - cell["esmall"][m]
            ratio = denc / zspread if zspread else float("inf")
            q2[m] = {"zspread": zspread, "denc": denc, "ratio": ratio}
            verdict = ("encoder ABOVE noise" if abs(denc) > NOISE[m]
                       else "encoder inside noise")
            P(f"{m:<11}{zspread:>11.4f}{NOISE[m]:>10.4f}{denc:>13.4f}"
              f"{ratio:>9.1f}x{verdict:>22}")
        else:
            P(f"{m:<11}{zspread:>11.4f}{NOISE[m]:>10.4f}{'n/a':>13}{'n/a':>10}")

    # ---------- Q3 unit-cost efficiency ----------
    P("\n### Q3  UNIT-COST EFFICIENCY  (E-small -> E-large)\n")
    if {"esmall", "elarge"} <= set(have):
        dpar = cell["elarge"]["params_M"] - cell["esmall"]["params_M"]
        dfl = cell["elarge"]["flops_G"] - cell["esmall"]["flops_G"]
        P(f"  Δparams = {dpar:.4f} M     ΔFLOPs = {dfl:.3f} G\n")
        P(f"{'metric':<11}{'Δ total':>11}{'per +0.01M params':>19}{'per +0.1 GFLOP':>17}")
        P("-" * 96)
        for m in METRICS:
            d = cell["elarge"][m] - cell["esmall"][m]
            pp = d / (dpar / 0.01) if dpar else 0
            pf = d / (dfl / 0.1) if dfl else 0
            P(f"{m:<11}{d:>+11.4f}{pp:>+19.5f}{pf:>+17.5f}")

    # ---------- Q4 saturation ----------
    P("\n### Q4  ENCODER SENSITIVITY / SATURATION\n")
    if len(have) == 3:
        above = []
        for m in METRICS:
            d_sm_base = cell["ebase"][m] - cell["esmall"][m]
            d_base_lg = cell["elarge"][m] - cell["ebase"][m]
            small_step_real = abs(d_sm_base) > NOISE[m]
            large_step_real = abs(d_base_lg) > NOISE[m]
            P(f"  {m:<11} small→base {d_sm_base:+.4f} "
              f"({'real' if small_step_real else 'noise'}),  "
              f"base→large {d_base_lg:+.4f} "
              f"({'real' if large_step_real else 'noise'})")
            if small_step_real:
                above.append(m)
        P("")
        if above:
            P(f"  Encoder capacity moves {len(above)}/{len(METRICS)} metrics "
              f"beyond noise on the small→base step.")
        else:
            P("  No metric moves beyond noise across the encoder sweep.")
        if {"elarge", "ebase"} <= set(have):
            sat = all(abs(cell["elarge"][m] - cell["ebase"][m]) <= NOISE[m]
                      for m in METRICS)
            if sat:
                P("  -> E-large ≈ E-base on every metric: SATURATION near the "
                  "baseline encoder.")
            else:
                mover = [m for m in METRICS
                         if abs(cell["elarge"][m] - cell["ebase"][m]) > NOISE[m]]
                P(f"  -> E-large still improves {mover}: no saturation yet at "
                  "E-large.")
    else:
        P("  need all three cells")

    out = os.path.join(OUTD, "exp3A_analysis.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
