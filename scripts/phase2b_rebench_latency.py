#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoritative latency re-benchmark for the Phase-2-B Z sweep.

WHY THIS EXISTS
---------------
`evaluation/evaluate_baseline.py` profiles latency via
`_profile_ours(model, (1,3,640,640), device, 3)` -> `latency_ms(warmup=10, reps=3)`.
Ten warm-up forwards (~70 ms of work) is nowhere near enough to ramp an idle
GPU out of its low-clock DVFS state, and 3 samples cannot support a P95.

Observed consequence (2026-09-03): the SAME z=16 checkpoint (188854 params,
1.0597 GFLOPs) measured
    - p50 2.495 ms / 395 FPS  (smoke test, run right after training = hot clocks)
    - p50 7.256 ms / 139 FPS  (sweep, first eval after the GPU idled = cold clocks)
while the long-running accuracy pass (`eval_fwd_ms`, 10k images) stayed at
6.06-6.56 ms in both. Bigger models appeared 3x FASTER than smaller ones, which
is physically impossible and would have produced a fabricated finding.

This script measures latency the way it should be measured:
  1. optional COLD pass  - measured before any ramp, to quantify the artifact
  2. CLOCK RAMP          - continuous forwards until >= ramp_seconds elapsed
  3. WARMUP              - >= warmup timed forwards
  4. MEASURE             - reps timed forwards, CUDA-synced each iteration
and reports the SM clock before/after so the ramp is verifiable.

Usage:
  gpu_env/bin/python scripts/phase2b_rebench_latency.py \
      --ckpt-glob 'experiments/phase2b/zsweep_z*/checkpoint.pt' \
      --out experiments/phase2b/zsweep_latency.csv \
      --ramp-seconds 5 --warmup 100 --reps 300 --cold
"""
import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def sm_clock_mhz():
    """Current SM clock in MHz, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def build(checkpoint):
    from evaluation.evaluate_baseline import build_ours
    from models.static_model import StaticMultiTaskModel
    model, _ = build_ours(StaticMultiTaskModel.__name__, checkpoint, "cuda")
    return model.to("cuda").eval()


def timed_passes(model, x, n):
    """Run n CUDA-synced forwards, return list of per-pass milliseconds."""
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def summarize(ts):
    ts = sorted(ts)
    mean = statistics.mean(ts)
    return {
        "mean_ms": mean,
        "p50_ms": ts[len(ts) // 2],
        "p95_ms": ts[min(int(len(ts) * 0.95), len(ts) - 1)],
        "min_ms": ts[0],
        "max_ms": ts[-1],
        "fps": 1000.0 / mean,
        "n": len(ts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob", required=True)
    ap.add_argument("--out", default="experiments/phase2b/zsweep_latency.csv")
    ap.add_argument("--input", type=int, default=640)
    ap.add_argument("--ramp-seconds", type=float, default=5.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--cold", action="store_true",
                    help="also measure BEFORE the clock ramp, to quantify the cold-clock artifact")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if not ckpts:
        print(f"[rebench] no checkpoints match {args.ckpt_glob}", file=sys.stderr)
        return 1
    # order by the numeric Z in the path so the CSV reads z16, z32, ...
    def zkey(p):
        try:
            return int(os.path.basename(os.path.dirname(p)).split("_z")[1])
        except Exception:
            return 0
    ckpts.sort(key=zkey)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    x = torch.randn(1, 3, args.input, args.input).cuda()
    rows = []

    for ck in ckpts:
        model = build(ck)
        row = {"checkpoint": ck}

        if args.cold:
            c_pre = sm_clock_mhz()
            cold = summarize(timed_passes(model, x, args.reps))
            c_post = sm_clock_mhz()
            row.update({
                "cold_sm_clk_pre_mhz": c_pre,
                "cold_sm_clk_post_mhz": c_post,
                "cold_p50_ms": round(cold["p50_ms"], 3),
                "cold_p95_ms": round(cold["p95_ms"], 3),
                "cold_fps": round(cold["fps"], 2),
            })

        # --- clock ramp: keep the GPU busy until ramp_seconds have elapsed ---
        ramp_start = time.perf_counter()
        with torch.no_grad():
            while time.perf_counter() - ramp_start < args.ramp_seconds:
                model(x)
            torch.cuda.synchronize()
        clk_hot = sm_clock_mhz()

        ts = timed_passes(model, x, args.warmup)      # warm-up (discarded)
        _ = ts
        ts = timed_passes(model, x, args.reps)        # measured
        s = summarize(ts)
        row.update({
            "sm_clk_mhz": clk_hot,
            "p50_ms": round(s["p50_ms"], 3),
            "p95_ms": round(s["p95_ms"], 3),
            "mean_ms": round(s["mean_ms"], 3),
            "min_ms": round(s["min_ms"], 3),
            "max_ms": round(s["max_ms"], 3),
            "fps": round(s["fps"], 2),
            "reps": s["n"],
        })
        rows.append(row)
        print("[rebench] {}  p50={}ms p95={}ms fps={} sm_clk={}MHz{}".format(
            os.path.basename(os.path.dirname(ck)), row["p50_ms"], row["p95_ms"],
            row["fps"], row["sm_clk_mhz"],
            "  (cold p50={}ms)".format(row["cold_p50_ms"]) if args.cold else ""),
            flush=True)
        del model
        torch.cuda.empty_cache()

    cols = ["checkpoint", "p50_ms", "p95_ms", "mean_ms", "min_ms", "max_ms",
            "fps", "reps", "sm_clk_mhz"]
    if args.cold:
        cols += ["cold_p50_ms", "cold_p95_ms", "cold_fps",
                 "cold_sm_clk_pre_mhz", "cold_sm_clk_post_mhz"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[rebench] wrote {args.out}  ({len(rows)} checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
