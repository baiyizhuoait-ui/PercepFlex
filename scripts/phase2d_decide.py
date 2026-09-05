#!/usr/bin/env python3
"""Phase 2-D · Statistical decision analysis.

Answers, from data only (no conclusion-shopping):
  Q1 Was the training budget masking a Z effect?
      -> compare the BUDGET effect size against the Z effect size.
  Q2 Is Z a real bottleneck? -> which tasks respond to z at all.
  Q3 Which task is most z-sensitive?
  Q4 Which z is the deployment pick (accuracy vs params/FLOPs)?
  Q5 Keep searching the knee?

Noise floor comes from Phase 2-C Exp A (3 seeds per z @ 4ep). That is the only
multi-seed dataset available; using it as a proxy for 20ep noise is CONSERVATIVE
(more converged runs are usually less noisy, so we over-estimate the floor).
"""
import csv, io, os
from statistics import mean, stdev

ROOT = "/home/mycode/ai_study/trac"
OUTD = os.path.join(ROOT, "experiments", "phase2d")
METRICS = ["mAP50", "da_mIoU", "lane_mIoU", "mAP50_95", "da_fg", "lane_fg"]

def rd(p):
    return list(csv.DictReader(io.open(p, encoding="utf-8"))) if os.path.exists(p) else []


def main():
    # ---- load budget table (9 cells) ----
    rows = []
    for r in rd(os.path.join(ROOT, "experiments/phase2b/zsweep_results.csv")):
        if r.get("seed") == "0" and r.get("epochs") == "4" and r["z"] in ("16", "32", "128"):
            rows.append({"z": int(r["z"]), "epochs": 4,
                         **{m: float(r[m]) for m in METRICS},
                         "params_M": float(r["params_M"]), "flops_G": float(r["flops_G"])})
    for r in rd(os.path.join(OUTD, "expD_budget.csv")):
        try:
            rows.append({"z": int(r["z"]), "epochs": int(r["epochs"]),
                         **{m: float(r[m]) for m in METRICS},
                         "params_M": float(r["params_M"]), "flops_G": float(r["flops_G"])})
        except (ValueError, KeyError):
            pass

    cell = {(r["z"], r["epochs"]): r for r in rows}
    ZS, EPS = [16, 32, 128], [4, 10, 20]

    # ---- noise floor from Phase 2-C Exp A (3 seeds @ 4ep) ----
    expA = rd(os.path.join(ROOT, "experiments/phase2c/expA_multiseed.csv"))
    # pull seed0 from the zsweep table so all three seeds are represented
    seed0 = {z: cell[(z, 4)] for z in ZS if (z, 4) in cell}
    noise = {}
    for m in METRICS:
        vals_by_z = {}
        for z in ZS:
            vs = []
            if z in seed0:
                vs.append(seed0[z][m])
            for r in expA:
                if int(r["z"]) == z:
                    vs.append(float(r[m]))
            if len(vs) >= 2:
                vals_by_z[z] = vs
        sds = [stdev(v) for v in vals_by_z.values() if len(v) > 1]
        pooled = (sum(s * s for s in sds) / len(sds)) ** 0.5 if sds else float("nan")
        noise[m] = pooled

    L = []
    P = L.append
    P("=" * 92)
    P("PHASE 2-D DECISION ANALYSIS")
    P("=" * 92)

    # ---------- Q1: budget effect vs z effect ----------
    P("\n### Q1  BUDGET EFFECT vs Z EFFECT  (spread = max - min across the 3 z's)\n")
    P(f"{'metric':<12} {'budget 4->20 (z16)':>20} {'z-spread@4':>13} {'z-spread@10':>13} "
      f"{'z-spread@20':>13} {'noise_floor':>12}")
    P("-" * 92)
    q1 = {}
    for m in METRICS:
        b_effect = cell[(16, 20)][m] - cell[(16, 4)][m]
        # also average across z for robustness
        b_eff_all = mean(cell[(z, 20)][m] - cell[(z, 4)][m] for z in ZS)
        sp = {ep: max(cell[(z, ep)][m] for z in ZS) - min(cell[(z, ep)][m] for z in ZS)
              for ep in EPS}
        q1[m] = {"budget_z16": b_effect, "budget_avg": b_eff_all, "spread": sp, "noise": noise[m]}
        P(f"{m:<12} {b_eff_all:>20.4f} {sp[4]:>13.4f} {sp[10]:>13.4f} "
          f"{sp[20]:>13.4f} {noise[m]:>12.4f}")

    P("\n  ratio  |budget effect| / z-spread@20   (>1 means budget dominates)")
    for m in METRICS:
        d = q1[m]
        r = abs(d["budget_avg"]) / d["spread"][20] if d["spread"][20] else float("inf")
        P(f"    {m:<12} {r:>8.1f}x")

    P("\n  z-spread@20 vs noise floor  (spread <= noise => z effect indistinguishable)")
    for m in METRICS:
        d = q1[m]
        verdict = "INSIDE noise" if d["spread"][20] <= d["noise"] else "above noise"
        P(f"    {m:<12} spread={d['spread'][20]:.4f}  noise={d['noise']:.4f}  -> {verdict}")

    # ---------- Q2 / Q3: per-task z sensitivity at 20ep ----------
    P("\n### Q2/Q3  PER-TASK Z SENSITIVITY @ 20ep  (relative spread = spread / mean)\n")
    P(f"{'metric':<12} {'z=16':>10} {'z=32':>10} {'z=128':>10} {'spread':>10} "
      f"{'rel%':>8} {'noise':>10} {'verdict':>18}")
    P("-" * 92)
    for m in METRICS:
        v = [cell[(z, 20)][m] for z in ZS]
        sp = max(v) - min(v)
        rel = 100.0 * sp / mean(v)
        verdict = "INSIDE noise" if sp <= noise[m] else "above noise"
        P(f"{m:<12} {v[0]:>10.4f} {v[1]:>10.4f} {v[2]:>10.4f} {sp:>10.4f} "
          f"{rel:>7.2f}% {noise[m]:>10.4f} {verdict:>18}")

    P("\n  NOTE (from docs/PHASE2D_BOTTLENECK_AUDIT.md): in R0 the detection head reads")
    P("  encoder F2/F3/F4 DIRECTLY and never sees Z. Only da/lane consume Z. So any")
    P("  mAP50 spread is gradient coupling + run noise, NOT a Z-capacity effect.")

    # ---------- Q4: Pareto ----------
    P("\n### Q4  PARETO @ 20ep  (accuracy vs params / FLOPs)\n")
    P(f"{'z':>5} {'params_M':>10} {'flops_G':>10} {'mAP50':>10} {'da_mIoU':>10} {'lane_mIoU':>11}")
    P("-" * 92)
    for z in ZS:
        r = cell[(z, 20)]
        P(f"{z:>5} {r['params_M']:>10.3f} {r['flops_G']:>10.3f} {r['mAP50']:>10.4f} "
          f"{r['da_mIoU']:>10.4f} {r['lane_mIoU']:>11.4f}")

    P("\n  pairwise deltas (row - col), positive = row is better")
    for i, a in enumerate(ZS):
        for b in ZS[i + 1:]:
            ra, rb = cell[(a, 20)], cell[(b, 20)]
            dm = ra["mAP50"] - rb["mAP50"]
            dd = ra["da_mIoU"] - rb["da_mIoU"]
            dl = ra["lane_mIoU"] - rb["lane_mIoU"]
            dpar = (ra["params_M"] - rb["params_M"]) / rb["params_M"] * 100
            dfl = (ra["flops_G"] - rb["flops_G"]) / rb["flops_G"] * 100
            P(f"    z{a} - z{b}:  mAP50 {dm:+.4f}  da {dd:+.4f}  lane {dl:+.4f} | "
              f"params {dpar:+.1f}%  flops {dfl:+.1f}%")

    # dominance check
    P("\n  dominance (a dominates b if a is <= on params AND flops AND >= on all metrics):")
    for i, a in enumerate(ZS):
        for b in ZS:
            if a == b:
                continue
            ra, rb = cell[(a, 20)], cell[(b, 20)]
            cheaper = (ra["params_M"] <= rb["params_M"] and ra["flops_G"] <= rb["flops_G"])
            better = all(ra[m] >= rb[m] for m in ("mAP50", "da_mIoU", "lane_mIoU"))
            if cheaper and better:
                P(f"    z{a} DOMINATES z{b}  (cheaper on params+flops, >= on all 3 metrics)")

    # ---------- Q5 ----------
    P("\n### Q5  VERDICT\n")
    primary = ["da_mIoU", "lane_mIoU"]  # the only metrics that actually read Z
    inside = sum(1 for m in METRICS if q1[m]["spread"][20] <= noise[m])
    shrinking = q1["da_mIoU"]["spread"][20] <= q1["da_mIoU"]["spread"][4]
    P(f"  metrics whose z-spread@20 is inside noise: {inside}/{len(METRICS)}")
    P(f"  da_mIoU spread shrank 4ep->20ep: {shrinking} "
      f"({q1['da_mIoU']['spread'][4]:.4f} -> {q1['da_mIoU']['spread'][20]:.4f})")
    P(f"  budget effect / z-spread@20 (mAP50): "
      f"{abs(q1['mAP50']['budget_avg'])/q1['mAP50']['spread'][20]:.1f}x")
    P("")
    if inside == len(METRICS) and not shrinking:
        P("  BRANCH A -> z effect EMERGED with budget: continue capacity search.")
    else:
        P("  BRANCH B -> z effect did NOT emerge; STOP searching the knee.")

    out = os.path.join(OUTD, "expD_decision_analysis.txt")
    io.open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
