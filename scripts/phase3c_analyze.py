#!/usr/bin/env python
"""Phase 3C - fixed-budget capacity allocation analysis.

Reads every result CSV (3A holds z=16, 3B holds z=32/128, 3C holds the two new
budget-matching cells) and produces the full analysis. No number is transcribed
by hand; if a cell is missing it is simply absent from the tables.

Runs in a degraded-but-correct mode before the two new cells finish, so the
analysis code can be validated while training is still in flight.
"""
import csv, os, sys, itertools

ROOT = "/home/mycode/ai_study/trac"
CSVS = [os.path.join(ROOT, "experiments/phase3a/exp3A_encoder.csv"),
        os.path.join(ROOT, "experiments/phase3b/phase3B_encoder_z.csv"),
        os.path.join(ROOT, "experiments/phase3c/phase3C_new_cells.csv")]
OUTDIR = os.path.join(ROOT, "experiments/phase3c")
ANALYSIS = os.path.join(OUTDIR, "phase3C_analysis.txt")
MATRIX = os.path.join(OUTDIR, "phase3C_budget_matrix.csv")
PARETO = os.path.join(OUTDIR, "phase3C_pareto.csv")
ALLOC = os.path.join(OUTDIR, "phase3C_allocation.csv")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
TASK_OF = {"mAP50": "detection", "mAP50_95": "detection",
           "da_mIoU": "DA", "da_fg": "DA",
           "lane_mIoU": "lane", "lane_fg": "lane"}
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}
TASKS = ["detection", "DA", "lane"]
ENC_ORDER = {"esmall": 0.0, "midL": 0.835, "ebase": 1.0, "midM": 1.75, "elarge": 2.0}
Z_RANK = {16: 0, 32: 1, 128: 2}
BUDGETS = [("Budget-L", 0.19), ("Budget-M", 0.29), ("Budget-H", 0.39)]
TOL = 0.05

w = []
def p(s=""):
    w.append(s); print(s)

def nm(r): return "%s_z%d" % (r["encoder"], r["z"])

def load():
    seen, rows = set(), []
    for path in CSVS:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    r["params_M"] = float(r["params_M"]); r["flops_G"] = float(r["flops_G"])
                    r["z"] = int(r["z"])
                    for m in METRICS:
                        r[m] = float(r[m])
                except (ValueError, KeyError):
                    continue
                key = (r["encoder"], r["z"])
                if key in seen:
                    continue
                seen.add(key); rows.append(r)
    return rows

def layer_roles(members):
    s = sorted(members, key=lambda r: (-ENC_ORDER.get(r["encoder"], 99), Z_RANK.get(r["z"], 99)))
    roles = {}
    if len(s) == 1:
        roles[id(s[0])] = "single"
    elif len(s) == 2:
        roles[id(s[0])] = "encoder-heavy"; roles[id(s[1])] = "Z-heavy"
    else:
        roles[id(s[0])] = "encoder-heavy"; roles[id(s[-1])] = "Z-heavy"
        for r in s[1:-1]:
            roles[id(r)] = "balanced"
    return roles

def beyond(a, b, m):
    return abs(a[m] - b[m]) > NOISE[m]

rows = load()
os.makedirs(OUTDIR, exist_ok=True)

p("=" * 104)
p("PHASE 3C - FIXED-BUDGET CAPACITY ALLOCATION ANALYSIS")
p("=" * 104)
for c in CSVS:
    p("  %-70s %s" % (os.path.relpath(c, ROOT), "found" if os.path.exists(c) else "ABSENT"))
p("  cells loaded: %d" % len(rows))
p()
p("  Noise floor is an EXTERNAL reference (Phase 2-C, 3 seeds, 4 epochs).")
p("  Phase 3C is single-seed (seed=0) and does NOT estimate its own variance.")
p()

p("-" * 104)
p("### A. FULL CELL POOL")
p("-" * 104)
p("%-14s %9s %9s  %-14s %s" % ("cell", "params_M", "flops_G", "source", "origin"))
for r in sorted(rows, key=lambda x: x["params_M"]):
    src = r.get("source", "?")
    org = "3C-new" if r["encoder"].startswith("mid") else ("3A" if r["z"] == 16 else "3B")
    p("%-14s %9.4f %9.4f  %-14s %s" % (nm(r), r["params_M"], r["flops_G"], src, org))
p()

p("-" * 104)
p("### B. BUDGET LAYERS AND THE ALLOCATION TRIAD")
p("-" * 104)
layers = {}
for bname, target in BUDGETS:
    members = [r for r in rows if abs(r["params_M"] - target) / target <= 0.10]
    roles = layer_roles(members)
    layers[bname] = (target, members, roles)
    p()
    p("  %s  target %.2f M   (%d cells within +/-10%%)" % (bname, target, len(members)))
    if not members:
        p("     empty")
        continue
    base = min(m["params_M"] for m in members)
    p("    %-14s %9s %9s %9s   %s" % ("cell", "params", "flops", "vs cheapest", "role"))
    for r in sorted(members, key=lambda x: x["params_M"]):
        p("    %-14s %9.4f %9.4f   %+7.2f%%   %s" % (
            nm(r), r["params_M"], r["flops_G"],
            (r["params_M"] - base) / base * 100, roles[id(r)]))
    span = (max(m["params_M"] for m in members) - base) / base * 100
    p("    layer param span: %.2f%%   -> %s" % (
        span, "EQUAL-BUDGET OK" if span <= TOL * 100 else "EXCEEDS %.0f%% TOL" % (TOL * 100)))
p()

p("-" * 104)
p("### C. ALLOCATION COMPARISON WITHIN EACH BUDGET")
p("-" * 104)
alloc_rows = []
for bname, target in BUDGETS:
    target_v, members, roles = layers[bname]
    if len(members) < 2:
        p()
        p("  %s: only %d cell - no comparison" % (bname, len(members)))
        continue
    p()
    p("  %s (~%.2f M)  - all three allocations at the SAME parameter budget" % (bname, target_v))
    order = sorted(members, key=lambda r: ["encoder-heavy", "balanced", "Z-heavy", "single"].index(roles[id(r)]))
    p("    %-14s %-14s %9s %9s %9s %9s %9s %9s %9s" % (
        "cell", "role", "params", "flops", "mAP50", "mAP50-95", "da_mIoU", "da_fg", "lane_mIoU"))
    for r in order:
        p("    %-14s %-14s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f" % (
            nm(r), roles[id(r)], r["params_M"], r["flops_G"],
            r["mAP50"], r["mAP50_95"], r["da_mIoU"], r["da_fg"], r["lane_mIoU"]))
        alloc_rows.append(dict(budget=bname, cell=nm(r), role=roles[id(r)],
                               params_M=r["params_M"], flops_G=r["flops_G"],
                               **{m: r[m] for m in METRICS}))
    p()
    ref = [r for r in members if roles[id(r)] == "encoder-heavy"]
    if ref:
        ref = ref[0]
        p("    delta vs the encoder-heavy cell (%s):" % nm(ref))
        p("      %-14s %-14s %10s %10s %10s %10s %10s %10s" % (
            "cell", "role", "dParams", "dFLOPs", "d mAP50", "d da_mIoU", "d lane_mIoU", "d lane_fg"))
        for r in order:
            if id(r) == id(ref):
                continue
            p("      %-14s %-14s %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f %+10.4f" % (
                nm(r), roles[id(r)], r["params_M"] - ref["params_M"],
                r["flops_G"] - ref["flops_G"], r["mAP50"] - ref["mAP50"],
                r["da_mIoU"] - ref["da_mIoU"], r["lane_mIoU"] - ref["lane_mIoU"],
                r["lane_fg"] - ref["lane_fg"]))
p()

p("-" * 104)
p("### D. TASK-WISE RESULTS (never collapsed into one score)")
p("-" * 104)
for task in TASKS:
    ms = [m for m in METRICS if TASK_OF[m] == task]
    p()
    p("  --- %s ---" % task)
    for bname, target in BUDGETS:
        target_v, members, roles = layers[bname]
        if len(members) < 2:
            continue
        order = sorted(members, key=lambda r: ["encoder-heavy", "balanced", "Z-heavy", "single"].index(roles[id(r)]))
        p("    %s (~%.2f M)" % (bname, target_v))
        for m in ms:
            vals = "  ".join("%s=%.4f" % (roles[id(r)][:4], r[m]) for r in order)
            best = max(order, key=lambda r: r[m])
            spread = max(r[m] for r in order) - min(r[m] for r in order)
            verdict = "beyond noise" if spread > NOISE[m] else "within noise"
            p("      %-10s %s   best=%-9s spread=%.4f (noise %.4f) %s" % (
                m, vals, roles[id(best)], spread, NOISE[m], verdict))
p()

p("-" * 104)
p("### E. PARAMS VS FLOPS (equal params does NOT mean equal compute)")
p("-" * 104)
for bname, target in BUDGETS:
    target_v, members, roles = layers[bname]
    if len(members) < 2:
        continue
    order = sorted(members, key=lambda r: ["encoder-heavy", "balanced", "Z-heavy", "single"].index(roles[id(r)]))
    p()
    p("  %s: parameter span %.2f%% but FLOPs span %.1f%%" % (
        bname,
        (max(r["params_M"] for r in members) - min(r["params_M"] for r in members))
        / min(r["params_M"] for r in members) * 100,
        (max(r["flops_G"] for r in members) - min(r["flops_G"] for r in members))
        / min(r["flops_G"] for r in members) * 100))
    for r in order:
        p("    %-14s %-14s params %.4f  flops %.4f   FLOPs per 0.01M params %.4f" % (
            nm(r), roles[id(r)], r["params_M"], r["flops_G"],
            r["flops_G"] / (r["params_M"] * 100)))
p()

p("-" * 104)
p("### F. FIXED-BUDGET DOMINANCE")
p("-" * 104)
p("  A dominates B only if params <=, FLOPs <=, and all six metrics >=,")
p("  where a metric counts as a real advantage only beyond the noise floor.")
p()
dom_rows = []
found_any = False
for a, b in itertools.permutations(rows, 2):
    lo, hi = sorted([a["params_M"], b["params_M"]])
    if (hi - lo) / lo > TOL:
        continue
    cost_ok = (a["params_M"] <= b["params_M"] + 1e-9) and (a["flops_G"] <= b["flops_G"] + 1e-9)
    acc_ok = all(a[m] >= b[m] - 1e-9 for m in METRICS)
    real = [m for m in METRICS if beyond(a, b, m)]
    if cost_ok and acc_ok:
        found_any = True
        p("  %s DOMINATES %s" % (nm(a), nm(b)))
        p("    params %.4f vs %.4f | flops %.4f vs %.4f" % (
            a["params_M"], b["params_M"], a["flops_G"], b["flops_G"]))
        p("    metrics beyond noise: %s" % (", ".join(real) if real else "none (all within noise)"))
        p()
    dom_rows.append(dict(a=nm(a), b=nm(b), params_a=a["params_M"], params_b=b["params_M"],
                         flops_a=a["flops_G"], flops_b=b["flops_G"],
                         dominates="yes" if (cost_ok and acc_ok) else "no",
                         beyond_noise=";".join(real)))
if not found_any:
    p("  No strict dominance found among equal-budget pairs.")
    p("  (A pair is only tested when its parameter gap is within %.0f%%.)" % (TOL * 100))
p()

p("-" * 104)
p("### G. PARETO FRONTIER (all cells, cost = params AND flops)")
p("-" * 104)
front, dominated = [], []
for r in rows:
    dom = False
    for o in rows:
        if o is r:
            continue
        cheaper = (o["params_M"] <= r["params_M"] + 1e-4) and (o["flops_G"] <= r["flops_G"] + 1e-4)
        better = all(o[m] >= r[m] - 1e-4 for m in METRICS)
        strictly = (o["params_M"] < r["params_M"] - 1e-4) or (o["flops_G"] < r["flops_G"] - 1e-4) \
                   or any(o[m] > r[m] + 1e-4 for m in METRICS)
        if cheaper and better and strictly:
            dom = True
            break
    (dominated if dom else front).append(r)
p("  NON-DOMINATED (%d):" % len(front))
for r in sorted(front, key=lambda x: x["params_M"]):
    p("    %-14s params %.4f  flops %.4f  mAP50 %.4f  da %.4f  lane %.4f" % (
        nm(r), r["params_M"], r["flops_G"], r["mAP50"], r["da_mIoU"], r["lane_mIoU"]))
p()
p("  DOMINATED (%d):" % len(dominated))
for r in sorted(dominated, key=lambda x: x["params_M"]):
    p("    %-14s params %.4f  flops %.4f  mAP50 %.4f" % (
        nm(r), r["params_M"], r["flops_G"], r["mAP50"]))
p()

p("-" * 104)
p("### H. ALLOCATION EFFICIENCY (per task, never mixed)")
p("-" * 104)
p("  In an equal-budget comparison dParams ~ 0, so a per-parameter ratio")
p("  DIVERGES and must not be reported. The honest statement is the absolute")
p("  gain obtained by moving the SAME budget. FLOPs do differ, so per-FLOP")
p("  ratios are meaningful and are reported below.")
p()
for bname, target in BUDGETS:
    target_v, members, roles = layers[bname]
    eh = [r for r in members if roles[id(r)] == "encoder-heavy"]
    zh = [r for r in members if roles[id(r)] == "Z-heavy"]
    if not eh or not zh:
        continue
    eh, zh = eh[0], zh[0]
    p()
    p("  %s: encoder-heavy %s vs Z-heavy %s" % (bname, nm(eh), nm(zh)))
    p("    dParams %+.4f M   dFLOPs %+.4f G" % (
        eh["params_M"] - zh["params_M"], eh["flops_G"] - zh["flops_G"]))
    for task in TASKS:
        ms = [m for m in METRICS if TASK_OF[m] == task]
        ds = [eh[m] - zh[m] for m in ms]
        mean_d = sum(ds) / len(ds)
        dfl = eh["flops_G"] - zh["flops_G"]
        real = any(abs(d) > NOISE[m] for d, m in zip(ds, ms))
        if not real:
            note = "difference within noise - no efficiency claim is made"
        elif dfl < -1e-6 and mean_d > 0:
            note = "encoder-heavy is BOTH more accurate AND cheaper -> it wins on both axes; a ratio would be misleading"
        elif dfl > 1e-6 and mean_d > 0:
            note = "encoder-heavy more accurate but costs %+.3f G -> per 0.1 GFLOPs %+.4f" % (
                dfl, mean_d / (dfl * 10))
        elif dfl < -1e-6 and mean_d < 0:
            note = "encoder-heavy cheaper but less accurate -> per 0.1 GFLOPs %+.4f" % (
                mean_d / (dfl * 10))
        else:
            note = "encoder-heavy worse on both axes"
        p("    %-10s mean delta %+.4f   dFLOPs %+.4f G   %s" % (task, mean_d, dfl, note))
p()

p("-" * 104)
p("### I. TASK-SPECIFIC CAPACITY DEMAND")
p("-" * 104)
p("  Phase 3B prior: detection wants encoder, DA saturated, lane wants Z.")
p("  Tested here under a FIXED budget, not assumed.")
p()
for task in TASKS:
    ms = [m for m in METRICS if TASK_OF[m] == task]
    score = {"encoder-heavy": 0, "balanced": 0, "Z-heavy": 0, "single": 0}
    details, decided, undecided = [], 0, 0
    for bname, target in BUDGETS:
        target_v, members, roles = layers[bname]
        if len(members) < 3:
            continue
        for m in ms:
            mx = max(r[m] for r in members)
            spread = mx - min(r[m] for r in members)
            if spread <= NOISE[m]:
                undecided += 1
                details.append("    %-10s %-10s spread %.4f <= noise %.4f -> NO allocation wins"
                               % (bname, m, spread, NOISE[m]))
                continue
            tie = [r for r in members if mx - r[m] <= NOISE[m]]
            roles_in = sorted(set(roles[id(r)] for r in tie))
            worst = min(members, key=lambda r: r[m])
            for rl in roles_in:
                score[rl] = score.get(rl, 0) + 1
            decided += 1
            details.append("    %-10s %-10s best (statistical tie): %-30s worst: %s [%s]"
                           % (bname, m, ",".join(roles_in), nm(worst), roles[id(worst)]))
    p()
    p("  %s   (%d metric-layer cells decided, %d within noise)" % (task, decided, undecided))
    for d in details:
        p(d)
    tot = sum(score.values())
    if tot:
        p("    -> wins: %s" % ", ".join(
            "%s x%d" % (k, v) for k, v in sorted(score.items(), key=lambda kv: -kv[1]) if v))
p()

p("-" * 104)
p("### J. STOPPING CONDITION (A / B / C / D)")
p("-" * 104)
det_layers, lane_layers = 0, 0
for bname, target in BUDGETS:
    target_v, members, roles = layers[bname]
    if len(members) < 3:
        continue
    d = [r for r in members if roles[id(r)] == "encoder-heavy"]
    z = [r for r in members if roles[id(r)] == "Z-heavy"]
    if not d or not z:
        continue
    d, z = d[0], z[0]
    if (d["mAP50"] - z["mAP50"]) > NOISE["mAP50"]:
        det_layers += 1
    if (z["lane_fg"] - d["lane_fg"]) > NOISE["lane_fg"]:
        lane_layers += 1
n_multi = sum(1 for b, t in BUDGETS if len(layers[b][1]) >= 3)
p("  budget layers with a full triad        : %d" % n_multi)
p("  layers where encoder-heavy wins detection beyond noise : %d" % det_layers)
p("  layers where Z-heavy wins lane beyond noise            : %d" % lane_layers)
p()
if det_layers >= 1 and lane_layers >= 1:
    rule = "B"
    p("  ==> RULE B: encoder-heavy wins detection, Z-heavy wins lane.")
    p("      Do NOT declare a single allocation winner. There is a genuine")
    p("      task-specific capacity allocation problem.")
elif det_layers >= 1:
    rule = "A"
    p("  ==> RULE A: encoder-heavy is better under a fixed budget, and no")
    p("      Z-heavy win was observed on any task. Supports spending on encoder.")
else:
    rule = "D"
    p("  ==> RULE D: differences do not clearly exceed the noise floor.")
p()
p("  (rule chosen: %s)" % rule)

with open(ANALYSIS, "w") as f:
    f.write("\n".join(w) + "\n")
with open(MATRIX, "w") as f:
    cols = ["cell", "encoder", "z", "params_M", "flops_G"] + METRICS + ["source"]
    f.write(",".join(cols) + "\n")
    for r in sorted(rows, key=lambda x: x["params_M"]):
        f.write(",".join([nm(r), r["encoder"], str(r["z"]),
                          "%.4f" % r["params_M"], "%.4f" % r["flops_G"]] +
                         ["%.4f" % r[m] for m in METRICS] + [r.get("source", "?")]) + "\n")
with open(PARETO, "w") as f:
    f.write("cell,params_M,flops_G,mAP50,da_mIoU,lane_mIoU,status\n")
    for r in sorted(rows, key=lambda x: x["params_M"]):
        f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n" % (
            nm(r), r["params_M"], r["flops_G"], r["mAP50"], r["da_mIoU"], r["lane_mIoU"],
            "non-dominated" if r in front else "dominated"))
with open(ALLOC, "w") as f:
    f.write("budget,cell,role,params_M,flops_G," + ",".join(METRICS) + "\n")
    for r in alloc_rows:
        f.write("%s,%s,%s,%.4f,%.4f,%s\n" % (
            r["budget"], r["cell"], r["role"], r["params_M"], r["flops_G"],
            ",".join("%.4f" % r[m] for m in METRICS)))
print()
for f in (ANALYSIS, MATRIX, PARETO, ALLOC):
    print("written:", f)
