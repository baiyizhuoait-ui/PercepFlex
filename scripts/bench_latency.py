#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase-2 Step-9 latency/benchmark harness.

Reports Params / real FLOPs / latency (mean, P50, P95) / FPS / peak GPU mem,
under PyTorch eager and torch.compile. 640x640, bs=1. SCAM-P-style measurement
(>=warmup + >=reps forward passes, CUDA-synced).

Usage:
    gpu_env/bin/python scripts/bench_latency.py \
        --checkpoint experiments/phase2/A0_static_nokd/checkpoint.pt \
        [--model OursStatic|OursDynamic] [--widths 0.25,0.5,1.0] \
        [--compile] [--reps 300] [--warmup 100]
"""
import argparse
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from profiling.flops_real import count_flops  # noqa: E402


def build(checkpoint, model_cls="OursStatic"):
    from evaluation.evaluate_baseline import build_ours
    if model_cls == "OursDynamic":
        from models.adaptive_model import AdaptiveMultiTaskModel
        model, _ = build_ours(AdaptiveMultiTaskModel.__name__, checkpoint, "cuda")
        return model
    from models.static_model import StaticMultiTaskModel
    model, _ = build_ours(StaticMultiTaskModel.__name__, checkpoint, "cuda")
    return model


def bench(model, x, device, warmup, reps):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = sorted(ts)
    return {"mean_ms": statistics.mean(ts), "p50_ms": ts[len(ts) // 2],
            "p95_ms": ts[int(len(ts) * 0.95)], "fps": 1000.0 / statistics.mean(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="OursStatic", choices=["OursStatic", "OursDynamic"])
    ap.add_argument("--widths", default=None, help="comma widths e.g. 0.25,0.5,1.0 (dynamic only)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--input", default=640)
    args = ap.parse_args()

    device = "cuda"
    model = build(args.checkpoint, args.model).to(device).eval()
    n_param = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, args.input, args.input).to(device)

    widths = [float(w) for w in args.widths.split(",")] if args.widths else [None]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        flops = count_flops(model, x)
    peak = torch.cuda.max_memory_allocated() / 1e6

    print(f"model={args.model} params={n_param/1e6:.3f}M flops={flops/1e9:.3f}G peak_mem={peak:.0f}MiB")
    for w in widths:
        if w is not None:
            from models.router.dynamic_conv import set_width_recursive
            set_width_recursive(model, w)
            with torch.no_grad():
                flops_w = count_flops(model, x)
            tag = f"width={w}"
        else:
            flops_w, tag = flops, "full"
        if args.compile:
            cmodel = torch.compile(model, mode="reduce-overhead")
            r = bench(cmodel, x, device, args.warmup, args.reps)
            print(f"[{tag}] compile=True  flops={flops_w/1e9:.3f}G "
                  f"mean={r['mean_ms']:.2f} p50={r['p50_ms']:.2f} p95={r['p95_ms']:.2f} fps={r['fps']:.1f}")
        r = bench(model, x, device, args.warmup, args.reps)
        print(f"[{tag}] compile=False flops={flops_w/1e9:.3f}G "
              f"mean={r['mean_ms']:.2f} p50={r['p50_ms']:.2f} p95={r['p95_ms']:.2f} fps={r['fps']:.1f}")


if __name__ == "__main__":
    main()
