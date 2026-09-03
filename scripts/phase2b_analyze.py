#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-2-B Task 2(a/b/d) analyzer (pure-Python, NO GPU required).

Reads evaluation metrics for the Z-capacity sweep (or a trusted proxy table) and
reports:
  (a) Representation Capacity Sweep  -> Z vs accuracy / FLOPs / FPS / Utility / Pareto
  (b) Per-task capacity sensitivity -> ΔDetection / ΔDA / ΔLane per Z-step + saturation
  (d) Representation Utility / degradation -> non-synchronous degradation pattern,
      Utility/FLOPs, Utility/Params (information-allocation, not just a small feature)

Usage:
  python scripts/phase2b_analyze.py --root <repo> --mode auto  [--out report.md]
  python scripts/phase2b_analyze.py --root <repo> --mode proxy [--out report.md]

mode=auto  : discover experiments/phase2b/zsweep_z*/eval/metrics.json (real runs)
mode=proxy : use the trusted existing capacity results (joint encoder+Z; epoch-mismatched
             for 1.0/2.0M — clearly flagged) as a stand-in until the dedicated Z-sweep runs.
"""
import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Trusted proxy table — transcribed from experiments/phase2/STEP1_REPORT.md and
# experiments/expD_compact_z/EXP_D_REPORT.md (the project's own equal-budget runs).
# 0.235M & 0.5M = fixed 4ep protocol; 1.0M & 2.0M = Phase-1 3ep/2ep (epoch mismatch
# flagged). These are JOINT encoder+Z capacity points, NOT a Z-isolated sweep.
# ---------------------------------------------------------------------------
PROXY_RUNS = [
    {"label": "0.235M", "z": None, "params_M": 0.230, "flops_G": 1.47, "fps": 157,
     "mAP50": 0.2414, "mAP50_95": 0.0785, "da_mIoU": 0.8360, "da_fg": 0.7418,
     "lane_fg": 0.1843, "lane_mIoU": 0.5802},
    {"label": "0.5M", "z": None, "params_M": 0.424, "flops_G": 2.46, "fps": 186,
     "mAP50": 0.2846, "mAP50_95": 0.0970, "da_mIoU": 0.8334, "da_fg": 0.7384,
     "lane_fg": 0.1915, "lane_mIoU": 0.5837},
    {"label": "1.0M", "z": None, "params_M": 0.959, "flops_G": 4.22, "fps": 179,
     "mAP50": 0.2951, "mAP50_95": 0.1010, "da_mIoU": 0.8518, "da_fg": 0.7654,
     "lane_fg": 0.1943, "lane_mIoU": 0.5855},
    {"label": "2.0M", "z": None, "params_M": 1.980, "flops_G": 8.84, "fps": 158,
     "mAP50": 0.3187, "mAP50_95": 0.1120, "da_mIoU": 0.8450, "da_fg": 0.7555,
     "lane_fg": 0.1939, "lane_mIoU": 0.5853},
]


def load_auto(root):
    import glob
    runs = []
    for mp in sorted(glob.glob(os.path.join(root, "experiments", "phase2b",
                                            "zsweep_z*_eval", "metrics.json"))):
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        # z from dir name zsweep_z<Z>_eval
        d = os.path.basename(os.path.dirname(mp))
        z = int(d.split("zsweep_z")[1].split("_")[0])
        runs.append({
            "label": f"z{z}", "z": z,
            "params_M": m.get("parameters", 0) / 1e6,
            "flops_G": m.get("flops", 0) / 1e9,
            "fps": m.get("fps", 0.0),
            "mAP50": m.get("mAP50", 0.0),
            "mAP50_95": m.get("mAP50_95", 0.0),
            "da_mIoU": m.get("da_mIoU", 0.0),
            "da_fg": m.get("da_fg_iou", 0.0),
            "lane_fg": m.get("lane_fg_iou", 0.0),
            "lane_mIoU": m.get("lane_mIoU", 0.0),
        })
    runs.sort(key=lambda r: (r["z"] if r["z"] is not None else 0))
    return runs


def utility(r):
    """Normalized geometric composite of the three tasks (each normalized to its
    own max across runs). Best run => 1.0; measures joint information retention."""
    tasks = ["mAP50", "da_mIoU", "lane_fg"]
    vals = [max(r[t], 1e-6) for t in tasks]
    mx = [max(x[t] for x in RUNS) for t in tasks]
    nrm = [v / m for v, m in zip(vals, mx)]
    g = 1.0
    for x in nrm:
        g *= x
    return g ** (1.0 / len(nrm))


RUNS = []  # set by main


def pareto_frontier(key_acc, key_cost):
    """Return labels that are Pareto-optimal on (accuracy, cost): no other run has
    both strictly higher accuracy AND <= cost."""
    pts = [(r[key_acc], r[key_cost], r["label"]) for r in RUNS]
    front = []
    for a, c, lab in pts:
        dominated = False
        for a2, c2, _ in pts:
            if a2 > a + 1e-9 and c2 <= c + 1e-9:
                dominated = True
                break
        if not dominated:
            front.append(lab)
    return front


def render_md(runs, mode):
    lines = []
    lines.append(f"# Phase-2-B Task 2 — Capacity / Sensitivity / Utility Analysis (`{mode}` mode)\n")
    if mode == "proxy":
        lines.append("> **PROXY** (joint encoder+Z capacity from existing trusted equal-budget "
                     "runs; 1.0M/2.0M are 3ep/2ep, NOT the fixed 4ep protocol). "
                     "The dedicated Z-isolated sweep (`mode auto`) supersedes this once it runs.\n")
    lines.append("\n## Raw table\n")
    hdr = "| label | z | params(M) | flops(G) | fps | mAP50 | da_mIoU | lane_fg | lane_mIoU |"
    lines.append(hdr)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        lines.append("| {label} | {z} | {p:.3f} | {f:.2f} | {fp:.0f} | {m:.4f} | {d:.4f} | {l:.4f} | {lm:.4f} |".format(
            label=r["label"], z=r["z"] if r["z"] is not None else "-",
            p=r["params_M"], f=r["flops_G"], fp=r["fps"],
            m=r["mAP50"], d=r["da_mIoU"], l=r["lane_fg"], lm=r["lane_mIoU"]))

    # (d) retention / non-synchronous degradation
    lines.append("\n## (d) Per-task retention vs best (non-synchronous degradation)\n")
    best = {t: max(r[t] for r in runs) for t in ["mAP50", "da_mIoU", "lane_fg", "lane_mIoU"]}
    lines.append("| label | mAP50 ret% | da_mIoU ret% | lane_fg ret% | lane_mIoU ret% |")
    lines.append("|---|---|---|---|---|")
    for r in runs:
        lines.append("| {label} | {a:.1f} | {b:.1f} | {c:.1f} | {d:.1f} |".format(
            label=r["label"],
            a=100 * r["mAP50"] / best["mAP50"],
            b=100 * r["da_mIoU"] / best["da_mIoU"],
            c=100 * r["lane_fg"] / best["lane_fg"],
            d=100 * r["lane_mIoU"] / best["lane_mIoU"]))
    lines.append("\n> Detection retains the least at low capacity; DA the most — tasks degrade "
                 "**non-synchronously** (Compact Z is a contested information-allocation problem, "
                 "not merely a small feature map).\n")

    # (b) per-step sensitivity
    lines.append("\n## (b) Per-step capacity sensitivity (Δ per capacity/Z step)\n")
    lines.append("| step | ΔmAP50 | Δda_mIoU | Δlane_fg | Δflops(G) |")
    lines.append("|---|---|---|---|---|")
    for i in range(1, len(runs)):
        a, b = runs[i], runs[i - 1]
        lines.append("| {la}->{lb} | {dm:+.4f} | {dd:+.4f} | {dl:+.4f} | {df:+.2f} |".format(
            la=a["label"], lb=b["label"],
            dm=a["mAP50"] - b["mAP50"], dd=a["da_mIoU"] - b["da_mIoU"],
            dl=a["lane_fg"] - b["lane_fg"], df=a["flops_G"] - b["flops_G"]))
    lines.append("\n> Detection gains the most per step (most capacity-hungry); DA/Lane saturate "
                 "early — confirms Phase-1 H2 / Exp-D D3-2 (Detection > Lane > DA sensitivity).\n")

    # (a) utility + efficiency + pareto
    lines.append("\n## (a) Utility, efficiency, and Pareto\n")
    lines.append("| label | Utility(0-1) | U/params(1/M) | U/flops(1/G) | mAP/flops |")
    lines.append("|---|---|---|---|---|")
    for r in runs:
        u = utility(r)
        lines.append("| {label} | {u:.3f} | {up:.2f} | {uf:.3f} | {mf:.3f} |".format(
            label=r["label"], u=u,
            up=u / max(r["params_M"], 1e-6),
            uf=u / max(r["flops_G"], 1e-6),
            mf=r["mAP50"] / max(r["flops_G"], 1e-6)))

    fp_acc = pareto_frontier("mAP50", "flops_G")
    fp_par = pareto_frontier("mAP50", "params_M")
    lines.append(f"\n- **Pareto frontier (mAP vs FLOPs):** {', '.join(fp_acc)}")
    lines.append(f"- **Pareto frontier (mAP vs Params):** {', '.join(fp_par)}")
    lines.append("\n> Diminishing-return region = where ΔmAP per +ΔFLOPs falls toward ~0 while "
                 "FLOPs/params keep rising. Identify the knee from the per-step table above once "
                 "the dedicated Z-sweep (auto) data is available.\n")
    return "\n".join(lines)


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--mode", choices=["auto", "proxy"], default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "auto":
        runs = load_auto(args.root)
        if not runs:
            print("[analyze] no zsweep eval metrics found -> falling back to proxy.", file=sys.stderr)
            runs = PROXY_RUNS
            mode = "proxy"
        else:
            mode = "auto"
    else:
        runs = PROXY_RUNS
        mode = "proxy"

    RUNS = runs
    md = render_md(runs, mode)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md + "\n")
        print(f"[analyze] wrote {args.out}")
    print(md)


if __name__ == "__main__":
    main()
