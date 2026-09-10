#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6B statistics — factorial interaction + equal-budget comparison.

Inputs (repo-relative):
  experiments/phase6/phase6_round2_factorial.csv     4 cells, 4ep seed0
  experiments/phase6/phase6_equal_budget.csv         arms ref/A/B/C
  experiments/phase6/persize_<tag>/metrics.json      size-stratified det metrics

Output:
  experiments/phase6/phase6_round2_statistics.csv    tidy (block, entity, metric, value, unit, note)

Design notes
  * The 2x2 factorial MUST be read through the interaction term
    interaction = dCap(km) - dCap(old);  a main-effect-only reading is invalid.
  * Noise reference: the Phase 5 pre-registration fixed the 4-epoch mAP50 noise
    scale at 1x = 0.0146 (used for the dp2a/danc gates). Phase 6B reuses the same
    protocol, so the same scale is quoted for the 4ep screening block.
"""
import csv
import io
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P6 = os.path.join(ROOT, "experiments", "phase6")

NOISE_1X_MAP50 = 0.0146   # Phase 5 pre-registered 4-epoch detection noise scale

FACT_TAGS = {"z16_old": "e7_z16_old", "z16_km": "e7_z16_km",
             "z32_old": "e7_z32_old", "z32_km": "e7_z32_km"}
PERSIZE_TAGS = dict(FACT_TAGS, unif20="e8_unif20")

rows = []


def add(block, entity, metric, value, unit="", note=""):
    if isinstance(value, float):
        value = round(value, 5)
    rows.append(dict(block=block, entity=entity, metric=metric,
                     value=value, unit=unit, note=note))


def load_csv(p):
    with io.open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(p):
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8-sig") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# A. EXP-07 factorial: interaction on every metric we have
# --------------------------------------------------------------------------- #
fact = {r["cell"]: r for r in load_csv(os.path.join(P6, "phase6_round2_factorial.csv"))}
ps = {}
for cell, tag in FACT_TAGS.items():
    j = load_json(os.path.join(P6, "persize_" + tag, "metrics.json"))
    if j:
        ps[cell] = j

for cell in ("z16_old", "z16_km", "z32_old", "z32_km"):
    if cell in fact:
        for k in ("params_M", "flops_G"):
            add("EXP-07.raw", cell, k, float(fact[cell][k]))

METRICS_CSV = [("mAP50", "mAP50", "det"), ("mAP50_95", "mAP50_95", "det"),
               ("da_mIoU", "da_mIoU", "DA"), ("lane_mIoU", "lane_mIoU", "lane")]
METRICS_PS = [(("det_AP50_" + b), ("det_AP50_" + b), "det_AP50_" + b)
              for b in ("small", "medium", "large")]
METRICS_PS += [(("det_AP5095_" + b), ("det_AP5095_" + b), "det_AP95_" + b)
               for b in ("small", "medium", "large")]
METRICS_PS += [(("det_recall50_" + b), ("det_recall50_" + b), "det_rec50_" + b)
               for b in ("small", "medium", "large")]


def _y(cell, src, key):
    if src == "csv":
        return float(fact[cell][key])
    j = ps.get(cell)
    return float(j[key]) if (j and key in j) else None


def analyse(metric_name, key, src):
    vals = {}
    for cell in ("z16_old", "z16_km", "z32_old", "z32_km"):
        try:
            vals[cell] = _y(cell, src, key)
        except Exception:
            return
    if any(v is None for v in vals.values()):
        return
    d_old = vals["z32_old"] - vals["z16_old"]
    d_km = vals["z32_km"] - vals["z16_km"]
    inter = d_km - d_old
    add("EXP-07.capacity", metric_name, "dCap_old(z32-z16)", d_old)
    add("EXP-07.capacity", metric_name, "dCap_kmeans(z32-z16)", d_km)
    add("EXP-07.interaction", metric_name, "interaction", inter,
        note="dCap(km)-dCap(old); negative = opposite to H-34")
    add("EXP-07.main", metric_name, "main_capacity", (d_old + d_km) / 2.0)
    add("EXP-07.main", metric_name, "main_assignment",
        ((vals["z16_km"] - vals["z16_old"]) + (vals["z32_km"] - vals["z32_old"])) / 2.0)
    if metric_name == "mAP50":
        add("EXP-07.gate", metric_name, "interaction_in_noise_units",
            abs(inter) / NOISE_1X_MAP50, unit="x",
            note="phase5 pre-registered 4ep noise 1x = %.4f" % NOISE_1X_MAP50)


for name, key, _ in METRICS_CSV:
    analyse(name, key, "csv")
for _, key, name in METRICS_PS:
    analyse(name, key, "json")


# --------------------------------------------------------------------------- #
# B. EXP-08 equal-budget comparison
# --------------------------------------------------------------------------- #
eq = load_csv(os.path.join(P6, "phase6_equal_budget.csv"))
ARM = {"e8_ref_r2z16": "ref", "e8_B_spatial_l14f1": "B_spatial",
       "e8_C_asym_combo": "C_asym", "e8_unif_seed0": "A_uniform"}
arms = {}
for r in eq:
    a = ARM.get(r["cell"])
    if not a:
        continue
    arms.setdefault(a, []).append(r)

TASKS = [("det", "mAP50"), ("lane", "lane_mIoU"), ("DA", "da_mIoU"),
         ("det_95", "mAP50_95"), ("lane_fg", "lane_fg"), ("DA_fg", "da_fg")]

agg = {}
for a, rs in arms.items():
    agg[a] = {"n": len(rs), "params_M": float(rs[0]["params_M"]),
              "flops_G": float(rs[0]["flops_G"])}
    for label, col in TASKS:
        vs = [float(r[col]) for r in rs if r.get(col) not in (None, "", "NA")]
        if vs:
            agg[a][label] = st.mean(vs)
            agg[a][label + "_sd"] = st.stdev(vs) if len(vs) > 1 else 0.0

for a, d in sorted(agg.items()):
    add("EXP-08.arm", a, "n_seeds", d["n"])
    add("EXP-08.arm", a, "params_M", d["params_M"])
    add("EXP-08.arm", a, "flops_G", d["flops_G"])
    for label, _ in TASKS:
        if label in d:
            add("EXP-08.arm", a, label, d[label], note="mean")
            add("EXP-08.arm", a, label + "_sd", d[label + "_sd"], note="stdev")

ref = agg.get("ref", {})
for a, d in sorted(agg.items()):
    if a == "ref":
        continue
    gains = {}
    for label, _ in [("det", 0), ("lane", 0), ("DA", 0)]:
        if label in d and label in ref and ref[label]:
            g_abs = d[label] - ref[label]
            g_rel = g_abs / ref[label]
            gains[label] = g_rel
            add("EXP-08.gain", a, "d_" + label, g_abs, note="abs vs ref")
            add("EXP-08.gain", a, "d_" + label + "_rel", g_rel, unit="frac")
    if len(gains) == 3:
        util = st.mean(gains.values())
        dfl = d["flops_G"] - ref["flops_G"]
        dpar = d["params_M"] - ref["params_M"]
        d["utility_meanrel"] = util          # stored for the Pareto block below
        add("EXP-08.utility", a, "utility_meanrel", util, unit="frac",
            note="mean of relative gains over det/lane/DA")
        add("EXP-08.cost", a, "d_flops_G", dfl)
        add("EXP-08.cost", a, "d_params_M", dpar)
        add("EXP-08.cost", a, "d_flops_rel", dfl / ref["flops_G"], unit="frac",
            note="parity rule: |frac| <= 0.05")
        add("EXP-08.cost", a, "d_params_rel", dpar / ref["params_M"], unit="frac",
            note="parity rule: |frac| <= 0.05")
        # Only quote a cost-normalised utility when the cost delta is large
        # enough for the ratio to mean anything; B/C sit on top of the ref
        # params (dParams ~0.02%), so a per-params ratio there is 1/x noise.
        if abs(dfl) / ref["flops_G"] >= 0.05:
            add("EXP-08.cost", a, "utility_per_dFLOPs", util / dfl, unit="frac/G")
        else:
            add("EXP-08.cost", a, "utility_per_dFLOPs", "NA", unit="frac/G",
                note="dFLOPs < 5% of ref - ratio not interpretable")
        if abs(dpar) / ref["params_M"] >= 0.05:
            add("EXP-08.cost", a, "utility_per_dParams", util / dpar, unit="frac/M")
        else:
            add("EXP-08.cost", a, "utility_per_dParams", "NA", unit="frac/M",
                note="dParams < 5% of ref - ratio not interpretable")

# --------------------------------------------------------------------------- #
# C. B vs C -- the round's only strictly single-variable contrast
#    phase4b_l14f1_z16.yaml vs phase6_combo_danc_l14f1.yaml differ ONLY by the
#    `anchors:` block (verified by diffing the model blocks), so params
#    (0.2019M) and FLOPs (1.6391G) are identical.  This isolates anchor
#    assignment at ZERO inference cost.
# --------------------------------------------------------------------------- #
b_arm, c_arm = agg.get("B_spatial", {}), agg.get("C_asym", {})
for label, _ in TASKS:
    if label in b_arm and label in c_arm:
        delta = c_arm[label] - b_arm[label]
        sds = [x for x in (b_arm.get(label + "_sd"), c_arm.get(label + "_sd")) if x]
        pooled = (sum(s * s for s in sds) / len(sds)) ** 0.5 if sds else 0.0
        add("EXP-08.contrast_BC", label, "C_minus_B", delta,
            note="anchors-only change at identical 0.2019M/1.6391G")
        add("EXP-08.contrast_BC", label, "C_minus_B_sigma",
            (abs(delta) / pooled) if pooled else "NA", unit="x",
            note="pooled 3-seed sd (B n=3, C n=3)")

# Pareto over (utility, flops) and (utility, params) among non-ref arms
pts = []
for a, d in agg.items():
    if a == "ref" or "utility_meanrel" not in d:
        continue
    pts.append((a, d["utility_meanrel"], d["flops_G"], d["params_M"]))
for a, u, f, p in pts:
    dom_f = any((u2 >= u and f2 <= f and (u2 > u or f2 < f)) for a2, u2, f2, _ in pts if a2 != a)
    dom_p = any((u2 >= u and p2 <= p and (u2 > u or p2 < p)) for a2, u2, _, p2 in pts if a2 != a)
    add("EXP-08.pareto", a, "dominated_on_flops", int(dom_f),
        note="1 = some arm has >= utility with <= FLOPs")
    add("EXP-08.pareto", a, "dominated_on_params", int(dom_p))

out = os.path.join(P6, "phase6_round2_statistics.csv")
with io.open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["block", "entity", "metric", "value", "unit", "note"],
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("wrote %s  (%d rows)" % (out, len(rows)))
print()
print("--- EXP-07 interaction (headline) ---")
for r in rows:
    if r["block"] == "EXP-07.interaction":
        print("  %-18s %+0.5f" % (r["metric"], r["value"]))
print("--- EXP-08 utility ---")
for r in rows:
    if r["block"] == "EXP-08.utility":
        print("  %-12s %+0.4f%%" % (r["entity"], r["value"] * 100))
