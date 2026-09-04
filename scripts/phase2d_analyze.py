#!/usr/bin/env python3
"""Phase 2-D · Aggregation + loss-curve extraction.

Reads:
  - experiments/phase2c/expA_multiseed.csv   (4ep, 3 seeds × 3 zs; the 3-seed noise floor)
  - experiments/phase2b/zsweep_results.csv   (4ep, seed=0 × all 6 zs in the original sweep)
  - experiments/phase2d/expD_budget.csv      (10ep/20ep/4ep seed=0 budget cells; live)
  - experiments/phase2d/expD_z{Z}_e{E}/training_log.txt  (per-step loss)

Outputs (text + CSV, never asserts beyond what data says):
  - docs/PHASE2D_BUDGET_RESULTS.md   (markdown summary, never edited by hand)
  - experiments/phase2d/expD_loss_curves.csv   (epoch-summary loss: avg_loss per epoch)
  - experiments/phase2d/expD_summary.txt       (formatted table, also printed to stdout)

The script is read-only. It only writes into experiments/phase2d/ and docs/.

Usage:
  PY=/home/mycode/ai_study/gpu_env/bin/python
  $PY scripts/phase2d_analyze.py
"""
from __future__ import annotations
import csv, json, os, re, sys
from collections import defaultdict
from statistics import mean, stdev

ROOT = "/home/mycode/ai_study/trac"
OUTD = os.path.join(ROOT, "experiments", "phase2d")
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

# ---------- input readers ----------
def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# ---------- loss-curve extraction from training_log.txt ----------
LOG_RE = re.compile(
    r"^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s+st(?P<stage>\S+)\s+ep(?P<epoch>\d+)\s+"
    r"step\s+(?P<step>\d+)/(?P<steps>\d+).*?\|"
    r"\s+loss\s+(?P<loss>\S+)"
    r"\s+det\s+(?P<det>\S+)"
    r"\s+da\s+(?P<da>\S+)"
    r"\s+lane\s+(?P<lane>\S+)"
)

EPOCH_DONE_RE = re.compile(r"^\[stage\s+(\S+)\]\s+ep\s+(\d+)\s+DONE\s+avg_loss=([\d.]+)\s+saved\s+(\S+)")

def parse_log(path):
    """Return list of per-step dicts and per-epoch summary."""
    steps = []
    epochs = []
    if not os.path.exists(path):
        return steps, epochs
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = LOG_RE.search(line)
            if m:
                steps.append({
                    "time": m["time"],
                    "stage": m["stage"],
                    "epoch": int(m["epoch"]),
                    "step": int(m["step"]),
                    "steps": int(m["steps"]),
                    "loss": float(m["loss"]),
                    "det": float(m["det"]),
                    "da": float(m["da"]),
                    "lane": float(m["lane"]),
                })
            m2 = EPOCH_DONE_RE.match(line)
            if m2:
                epochs.append({
                    "stage": m2.group(1),
                    "epoch": int(m2.group(2)),
                    "avg_loss": float(m2.group(3)),
                    "ckpt": m2.group(4),
                })
    return steps, epochs

def extract_loss_curves(ep_values, z_values):
    """For each (z, ep) cell, read the training log and produce per-epoch loss summaries."""
    rows = []
    for z in z_values:
        for ep in ep_values:
            log = os.path.join(OUTD, f"expD_z{z}_e{ep}", "training_log.txt")
            steps, epochs = parse_log(log)
            if not steps and not epochs:
                continue
            # group by epoch
            by_ep = defaultdict(list)
            for s in steps:
                by_ep[s["epoch"]].append(s)
            for ep_no, items in sorted(by_ep.items()):
                rows.append({
                    "z": z,
                    "epochs_configured": ep,
                    "epoch": ep_no,
                    "n_steps_logged": len(items),
                    "mean_loss": round(mean(it["loss"] for it in items), 6),
                    "mean_det": round(mean(it["det"] for it in items), 6),
                    "mean_da": round(mean(it["da"] for it in items), 6),
                    "mean_lane": round(mean(it["lane"] for it in items), 6),
                    "final_step_loss": items[-1]["loss"],
                })
            # also include the explicit EPOCH_DONE avg
            for ed in epochs:
                if ed["epoch"] == ep:
                    rows.append({
                        "z": z, "epochs_configured": ep, "epoch": ep_no,
                        "n_steps_logged": "", "mean_loss": "", "mean_det": "",
                        "mean_da": "", "mean_lane": "",
                        "epoch_done_avg_loss": ed["avg_loss"],
                    })
    return rows

# ---------- main ----------
def main():
    # Read all 4ep seed=0 from zsweep_results.csv (for the 4ep budget cell)
    zsweep = read_csv(os.path.join(ROOT, "experiments/phase2b/zsweep_results.csv"))
    zsweep4 = [r for r in zsweep if r.get("seed") == "0" and r.get("epochs") == "4"
               and r["z"] in ("16", "32", "128")]
    # 4ep seed=1,2 from expA_multiseed (for the noise floor only)
    expA = read_csv(os.path.join(ROOT, "experiments/phase2c/expA_multiseed.csv"))
    # expD live CSV
    expD = read_csv(os.path.join(OUTD, "expD_budget.csv"))

    # merge 4ep seed=0 into the budget table
    merged = []
    for r in zsweep4:
        merged.append({
            "z": int(r["z"]), "epochs": 4, "params_M": float(r["params_M"]),
            "flops_G": float(r["flops_G"]), "fps": float(r["fps"]),
            "mAP50": float(r["mAP50"]), "mAP50_95": float(r["mAP50_95"]),
            "da_mIoU": float(r["da_mIoU"]), "da_fg": float(r["da_fg"]),
            "lane_fg": float(r["lane_fg"]), "lane_mIoU": float(r["lane_mIoU"]),
            "train_wall_min": "from_4ep", "peak_gpu_mem_mib": "from_4ep",
            "final_train_loss": "from_4ep", "seed": int(r["seed"]),
            "source": "zsweep_results.csv",
        })
    for r in expD:
        try:
            merged.append({
                "z": int(r["z"]), "epochs": int(r["epochs"]),
                "params_M": float(r["params_M"]), "flops_G": float(r["flops_G"]),
                "fps": float(r["fps"]),
                "mAP50": float(r["mAP50"]), "mAP50_95": float(r["mAP50_95"]),
                "da_mIoU": float(r["da_mIoU"]), "da_fg": float(r["da_fg"]),
                "lane_fg": float(r["lane_fg"]), "lane_mIoU": float(r["lane_mIoU"]),
                "train_wall_min": r.get("train_wall_min", ""),
                "peak_gpu_mem_mib": r.get("peak_gpu_mem_mib", ""),
                "final_train_loss": r.get("final_train_loss", ""),
                "seed": int(r["seed"]),
                "source": "expD_budget.csv",
            })
        except (ValueError, KeyError):
            continue

    # Print a compact table
    print("=" * 100)
    print("PHASE 2-D BUDGET TABLE  (z × epochs, seed=0)")
    print("=" * 100)
    hdr = ["z", "ep", "params", "flops", "fps",
           "mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_fg", "lane_mIoU",
           "wall_min", "mem_MiB", "final_loss", "src"]
    print(" ".join(f"{h:>10}" for h in hdr))
    sorted_merged = sorted(merged, key=lambda r: (r["epochs"], r["z"]))
    for r in sorted_merged:
        cells = [
            f"{r['z']:>10d}", f"{r['epochs']:>10d}",
            f"{r['params_M']:>10.3f}", f"{r['flops_G']:>10.3f}", f"{r['fps']:>10.2f}",
            f"{r['mAP50']:>10.4f}", f"{r['mAP50_95']:>10.4f}",
            f"{r['da_mIoU']:>10.4f}", f"{r['da_fg']:>10.4f}",
            f"{r['lane_fg']:>10.4f}", f"{r['lane_mIoU']:>10.4f}",
            f"{str(r['train_wall_min']):>10s}", f"{str(r['peak_gpu_mem_mib']):>10s}",
            f"{str(r['final_train_loss'])[:10]:>10s}", f"{r['source']:>10s}",
        ]
        print(" ".join(cells))
    print()

    # Loss curves
    print("LOSS CURVES (per-epoch averages from training_log.txt)")
    print("-" * 80)
    curves = extract_loss_curves([10, 20], [16, 32, 128])
    if curves:
        # Write CSV
        curve_csv = os.path.join(OUTD, "expD_loss_curves.csv")
        with open(curve_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
            w.writeheader()
            w.writerows(curves)
        print(f"Wrote {curve_csv}  ({len(curves)} rows)")
        # Print compact
        for r in curves[:6]:
            print(r)
        if len(curves) > 6:
            print(f"  ... ({len(curves)-6} more rows in CSV)")

    # Save merged table CSV
    merged_csv = os.path.join(OUTD, "expD_budget_merged.csv")
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sorted_merged[0].keys()) if sorted_merged else [])
        if sorted_merged:
            w.writeheader()
            w.writerows(sorted_merged)
    print(f"\nWrote {merged_csv}  ({len(sorted_merged)} rows)")

    # 4ep noise floor (from expA seeds 1+2, 3 z values)
    if expA:
        print("\n4ep NOISE FLOOR (Phase 2-C Exp A: seed 1+2, 3 zs)")
        by_z = defaultdict(list)
        for r in expA:
            by_z[int(r["z"])].append(r)
        for z in sorted(by_z):
            for met in ("mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_fg", "lane_mIoU"):
                vals = [float(r[met]) for r in by_z[z]]
                print(f"  z={z:>3d}  {met:>10s}  n={len(vals)}  mean={mean(vals):.4f}  std={stdev(vals) if len(vals)>1 else float('nan'):.4f}")

if __name__ == "__main__":
    main()
