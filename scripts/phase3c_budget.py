#!/usr/bin/env python
"""Phase 3C Step 1 - budget grouping + fairness check on the EXISTING cells.

Reads BOTH result CSVs (Phase 3A holds the z=16 column, Phase 3B holds z=32/128)
so the full 3x3 grid is reconstructed. No numbers are transcribed by hand.
"""
import csv, os, itertools

ROOT = "/home/mycode/ai_study/trac"
CSVS = [os.path.join(ROOT, "experiments/phase3a/exp3A_encoder.csv"),
        os.path.join(ROOT, "experiments/phase3b/phase3B_encoder_z.csv")]
OUT = os.path.join(ROOT, "experiments/phase3c/phase3C_budget_check.txt")

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
TASK_OF = {"mAP50": "detection", "mAP50_95": "detection",
           "da_mIoU": "DA", "da_fg": "DA",
           "lane_mIoU": "lane", "lane_fg": "lane"}
NOISE = {"mAP50": 0.0073, "mAP50_95": 0.0032, "da_mIoU": 0.0142,
         "da_fg": 0.0202, "lane_mIoU": 0.0021, "lane_fg": 0.0032}
TOL = 0.05
BUDGETS = [("Budget-L", 0.19), ("Budget-M", 0.29), ("Budget-H", 0.39)]
ENC_RANK = {"esmall": 0, "ebase": 1, "elarge": 2}
Z_RANK = {16: 0, 32: 1, 128: 2}

w = []
def p(s=""):
    w.append(s); print(s)

def name(r): return "%s_z%d" % (r["encoder"], r["z"])

def load():
    seen, rows = {}, []
    for path in CSVS:
        with open(path) as f:
            for r in csv.DictReader(f):
                r["params_M"] = float(r["params_M"]); r["flops_G"] = float(r["flops_G"])
                r["z"] = int(r["z"])
                for m in METRICS:
                    r[m] = float(r[m])
                key = (r["encoder"], r["z"])
                if key not in seen:
                    seen[key] = 1; rows.append(r)
    return rows

def layer_roles(members):
    """Allocation roles are defined RELATIVE to the layer, not globally."""
    s = sorted(members, key=lambda r: (-ENC_RANK[r["encoder"]], Z_RANK[r["z"]]))
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

rows = load()
os.makedirs(os.path.dirname(OUT), exist_ok=True)

p("=" * 100)
p("P3C-STEP1 - BUDGET GROUPING + FAIRNESS CHECK")
p("=" * 100)
for c in CSVS: p("  source: %s" % os.path.relpath(c, ROOT))
p("  cells loaded: %d   (expect 9 = 3 encoders x 3 widths)" % len(rows))
p("  like-for-like tolerance: %.0f%% of the smaller params" % (TOL * 100))
p()

p("-" * 100)
p("### A. FULL GRID BY PARAMETER COUNT")
p("-" * 100)
p("%-14s %9s %9s  %s" % ("cell", "params_M", "flops_G", "source"))
for r in sorted(rows, key=lambda x: x["params_M"]):
    p("%-14s %9.4f %9.4f  %s" % (name(r), r["params_M"], r["flops_G"], r.get("source", "?")))
p()

p("-" * 100)
p("### B. EVERY PAIR WITHIN %.0f%% PARAMS" % (TOL * 100))
p("-" * 100)
pairs = []
for a, b in itertools.combinations(rows, 2):
    lo, hi = sorted([a["params_M"], b["params_M"]])
    dpct = (hi - lo) / lo
    if dpct <= TOL:
        pairs.append((dpct, a, b))
pairs.sort(key=lambda t: t[0])
if not pairs:
    p("  NONE")
for dpct, a, b in pairs:
    heavy, light_ = (a, b) if ENC_RANK[a["encoder"]] > ENC_RANK[b["encoder"]] else (b, a)
    p()
    p("  %s  (larger encoder + narrower z)   vs   %s  (smaller encoder + wider z)"
      % (name(heavy), name(light_)))
    p("    dParams  %+.4f M  = %+.2f%% of smaller   [WITHIN %.0f%%]"
      % (heavy["params_M"] - light_["params_M"],
         (heavy["params_M"] - light_["params_M"]) / min(heavy["params_M"], light_["params_M"]) * 100,
         TOL * 100))
    p("    dFLOPs   %+.4f G  = %+.1f%% of smaller"
      % (heavy["flops_G"] - light_["flops_G"],
         (heavy["flops_G"] - light_["flops_G"]) / min(heavy["flops_G"], light_["flops_G"]) * 100))
    p("    %-10s %10s %10s %10s   %s" % ("metric", "enc-heavy", "Z-heavy", "delta", "read"))
    for m in METRICS:
        d = heavy[m] - light_[m]
        read = "beyond noise" if abs(d) > NOISE[m] else "within noise"
        p("    %-10s %10.4f %10.4f %+10.4f   %s (%s)" % (m, heavy[m], light_[m], d, read, TASK_OF[m]))
p()

p("-" * 100)
p("### C. BUDGET LAYERS  (cells within +/-10% of target)")
p("-" * 100)
layerinfo = {}
for bname, target in BUDGETS:
    members = [r for r in rows if abs(r["params_M"] - target) / target <= 0.10]
    roles = layer_roles(members)
    layerinfo[bname] = (target, members, roles)
    p()
    p("  %s  target %.2f M   -> %d cell(s)" % (bname, target, len(members)))
    if not members:
        p("     (empty - no comparison possible)")
        continue
    base = min(m["params_M"] for m in members)
    p("    %-14s %9s %9s %9s   %s" % ("cell", "params", "flops", "vs cheapest", "role in layer"))
    for r in sorted(members, key=lambda x: x["params_M"]):
        p("    %-14s %9.4f %9.4f   %+7.2f%%   %s" % (
            name(r), r["params_M"], r["flops_G"],
            (r["params_M"] - base) / base * 100, roles[id(r)]))
p()

p("-" * 100)
p("### D. TRIAD COMPLETENESS AND THE FAIRNESS GAP")
p("-" * 100)
for bname, target in BUDGETS:
    target_v, members, roles = layerinfo[bname]
    if not members:
        p()
        p("  %s: EMPTY" % bname); continue
    have = {}
    for r in members:
        have.setdefault(roles[id(r)], []).append(r)
    p()
    p("  %s (~%.2f M)" % (bname, target_v))
    for role in ["encoder-heavy", "balanced", "Z-heavy", "single"]:
        if role in have:
            for r in have[role]:
                p("    %-15s PRESENT  %-14s params %.4f  flops %.4f" % (
                    role, name(r), r["params_M"], r["flops_G"]))
    for role in ["encoder-heavy", "balanced", "Z-heavy"]:
        if role not in have and role != "single":
            p("    %-15s MISSING" % role)
    if "balanced" in have:
        b = have["balanced"][0]
        eh = have.get("encoder-heavy", [b])[0]
        zh = have.get("Z-heavy", [b])[0]
        for other, lbl in [(eh, "encoder-heavy"), (zh, "Z-heavy")]:
            if id(other) == id(b): continue
            dev = (b["params_M"] - other["params_M"]) / other["params_M"] * 100
            flag = "OK" if abs(dev) <= TOL * 100 else "EXCEEDS %.0f%% TOLERANCE" % (TOL * 100)
            p("      balanced %s is %+.2f%% vs %s -> %s" % (name(b), dev, lbl, flag))
p()

p("-" * 100)
p("### E. VERDICT")
p("-" * 100)
p("  like-for-like pairs already available within %.0f%%: %d" % (TOL * 100, len(pairs)))
for dpct, a, b in pairs:
    heavy = a if ENC_RANK[a["encoder"]] > ENC_RANK[b["encoder"]] else b
    light_ = b if heavy is a else a
    p("    %.2f%%   %s vs %s" % (dpct * 100, name(heavy), name(light_)))
p()
new_needed = 0
for bname, target in BUDGETS:
    target_v, members, roles = layerinfo[bname]
    have = set(roles.values())
    miss = [r for r in ["encoder-heavy", "balanced", "Z-heavy"] if r not in have]
    if not members:
        p("  %s: EMPTY -> needs 3 cells to form a triad" % bname)
        new_needed += 3
    elif miss:
        p("  %s: incomplete, missing %s" % (bname, ", ".join(miss)))
        new_needed += len(miss)
    else:
        p("  %s: triad COMPLETE from existing cells" % bname)
p()
p("  minimum NEW cells for full triads on all three budgets: %d" % new_needed)
p()
if len(pairs) >= 2:
    p("  => The two comparisons Phase 3C explicitly asks for (sec.5 items 1 and 2)")
    p("     already exist at 1-2%% param mismatch. The core research question can be")
    p("     answered from existing data with ZERO new training.")
    p("  => What is genuinely missing is a strictly budget-matched balanced leg")
    p("     (and Budget-H has only one cell). New training is only justified if the")
    p("     preliminary result is close to noise, or if a balanced winner is plausible.")

with open(OUT, "w") as f:
    f.write("\n".join(w) + "\n")
print(); print("written:", OUT)
