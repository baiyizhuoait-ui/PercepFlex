#!/usr/bin/env python3
"""Phase 3A · Allocation-efficiency ledger: encoder params vs Z params.

This is the analysis that actually answers the research question
"should limited parameters go to the encoder or to Z?".

Two spending records are compared:
  A) Phase 2-D  — spend ~+0.097M params on Z (z 16 -> 128), encoder FIXED
                  at baseline. 20ep, seed=0.
  B) Phase 3A   — spend +0.190M params on the ENCODER, z FIXED at 16.
                  20ep, seed=0.

For each we compute gain per +0.01M params and per +0.1 GFLOP.

Caveat stated honestly: the two sweeps start from different baselines
(Phase 2-D from 0.189M, Phase 3A from 0.101M), so marginal returns are not
strictly comparable at the margin. The like-for-like check is therefore
reported separately: compare two configs at ALMOST EQUAL total params.
"""
import csv, io, os

ROOT = "/home/mycode/ai_study/trac"
P3A = os.path.join(ROOT, "experiments", "phase3a", "exp3A_encoder.csv")
P2D = os.path.join(ROOT, "experiments", "phase2d", "expD_budget.csv")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}


def rd(p):
    return list(csv.DictReader(io.open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    p3a = {r["encoder"]: r for r in rd(P3A)}
    p2d = {(int(r["z"]), int(r["epochs"])): r for r in rd(P2D)}

    L = []
    P = L.append
    P("=" * 94)
    P("ALLOCATION EFFICIENCY LEDGER ·  encoder params  vs  Z params")
    P("=" * 94)

    # ---------- A: spending on Z (Phase 2-D, encoder fixed) ----------
    P("\n### A) Spend on Z    (Phase 2-D: z 16 -> 128, encoder FIXED at baseline, 20ep)\n")
    z16, z128 = p2d.get((16, 20)), p2d.get((128, 20))
    P(f"{'metric':<11}{'z=16':>10}{'z=128':>10}{'Δ':>10}"
      f"{'Δparams':>10}{'ΔFLOPs':>10}{'per 0.01M':>12}{'per 0.1GF':>11}{'real?':>8}")
    P("-" * 94)
    A_eff = {}
    if z16 and z128:
        dp = float(z128["params_M"]) - float(z16["params_M"])
        df = float(z128["flops_G"]) - float(z16["flops_G"])
        for m in METRICS:
            d = float(z128[m]) - float(z16[m])
            per_p = d / (dp / 0.01) if dp else 0.0
            per_f = d / (df / 0.1) if df else 0.0
            A_eff[m] = (per_p, per_f)
            P(f"{m:<11}{float(z16[m]):>10.4f}{float(z128[m]):>10.4f}{d:>+10.4f}"
              f"{dp:>10.4f}{df:>10.3f}{per_p:>+12.5f}{per_f:>+11.5f}"
              f"{('yes' if abs(d) > NOISE[m] else 'noise'):>8}")
    else:
        P("  [missing Phase 2-D 20ep rows]")

    # ---------- B: spending on encoder (Phase 3A, z fixed at 16) ----------
    P("\n### B) Spend on ENCODER   (Phase 3A: E-small -> E-large, z FIXED at 16, 20ep)\n")
    es, el = p3a.get("esmall"), p3a.get("elarge")
    P(f"{'metric':<11}{'E-small':>10}{'E-large':>10}{'Δ':>10}"
      f"{'Δparams':>10}{'ΔFLOPs':>10}{'per 0.01M':>12}{'per 0.1GF':>11}{'real?':>8}")
    P("-" * 94)
    B_eff = {}
    if es and el:
        dp = float(el["params_M"]) - float(es["params_M"])
        df = float(el["flops_G"]) - float(es["flops_G"])
        for m in METRICS:
            d = float(el[m]) - float(es[m])
            per_p = d / (dp / 0.01) if dp else 0.0
            per_f = d / (df / 0.1) if df else 0.0
            B_eff[m] = (per_p, per_f)
            P(f"{m:<11}{float(es[m]):>10.4f}{float(el[m]):>10.4f}{d:>+10.4f}"
              f"{dp:>10.4f}{df:>10.3f}{per_p:>+12.5f}{per_f:>+11.5f}"
              f"{('yes' if abs(d) > NOISE[m] else 'noise'):>8}")
    else:
        P("  [missing Phase 3A rows]")

    # ---------- C: ratio ----------
    P("\n### C) Efficiency ratio   encoder-spend / Z-spend\n")
    P(f"{'metric':<11}{'enc per 0.01M':>16}{'Z per 0.01M':>14}{'ratio (params)':>16}"
      f"{'ratio (FLOPs)':>15}")
    P("-" * 94)
    for m in METRICS:
        if m in A_eff and m in B_eff:
            bp, bf = B_eff[m]
            ap, af = A_eff[m]
            rp = (bp / ap) if abs(ap) > 1e-12 else float("inf")
            rf = (bf / af) if abs(af) > 1e-12 else float("inf")
            rs = f"{rp:>15.1f}x" if rp != float("inf") else f"{'inf':>16}"
            rfs = f"{rf:>14.1f}x" if rf != float("inf") else f"{'inf':>15}"
            P(f"{m:<11}{bp:>+16.5f}{ap:>+14.5f}{rs}{rfs}")

    # ---------- D: like-for-like at nearly equal total params ----------
    P("\n### D) LIKE-FOR-LIKE  (two configs at almost equal TOTAL params, 20ep, seed=0)\n")
    if z128 and el:
        pz, pe = float(z128["params_M"]), float(el["params_M"])
        fz, fe = float(z128["flops_G"]), float(el["flops_G"])
        P(f"{'':<11}{'base enc + z128':>18}{'E-large enc + z16':>20}{'delta':>12}")
        P("-" * 94)
        P(f"{'params M':<11}{pz:>18.4f}{pe:>20.4f}{(pe-pz)/pz*100:>11.1f}%")
        P(f"{'FLOPs G':<11}{fz:>18.3f}{fe:>20.3f}{(fe-fz)/fz*100:>11.1f}%")
        for m in METRICS:
            vz, ve = float(z128[m]), float(el[m])
            P(f"{m:<11}{vz:>18.4f}{ve:>20.4f}{ve-vz:>+12.4f}")
        P("")
        win_p = pe <= pz
        win_f = fe <= fz
        win_a = sum(1 for m in METRICS if float(el[m]) >= float(z128[m]))
        P(f"  E-large + z16 uses {'LESS' if win_p else 'MORE'} params "
          f"and {'LESS' if win_f else 'MORE'} FLOPs, and is >= on "
          f"{win_a}/{len(METRICS)} metrics.")
        if win_f and win_a >= 4:
            P("  -> Encoder-heavy + small-Z DOMINATES baseline-encoder + large-Z.")

    out = os.path.join(ROOT, "experiments", "phase3a", "exp3A_allocation_ledger.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
