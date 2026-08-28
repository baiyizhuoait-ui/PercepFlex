#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 smoke test: verify that every baseline model (B1-B3) can be
imported and forward-passed under the project environment, and report
Parameters / FLOPs for each preset.

Usage:
    python scripts/smoke_baselines.py [--device cpu|cuda] [--preset ...]
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.join(ROOT, "baselines")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def flops_of(model, x, device):
    """thop-based FLOPs (MACs*2 approx)."""
    try:
        from thop import profile
    except ImportError:
        return None
    model = model.to(device)
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    return 2 * macs  # FLOPs ~= 2 * MACs


def time_forward(model, x, device, reps=20, warmup=5):
    model = model.to(device).eval()
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    return dt


def build_twinlitenet(device):
    sys.path.insert(0, os.path.join(BASELINES, "TwinLiteNet"))
    from model.TwinLite import TwinLiteNet  # noqa: E402

    return TwinLiteNet().to(device)


def build_twinlitenetplus(preset, device):
    sys.path.insert(0, os.path.join(BASELINES, "TwinLiteNetPlus"))
    from model.model import TwinLiteNetPlus  # noqa: E402

    args = argparse.Namespace(config=preset)
    return TwinLiteNetPlus(args).to(device)


def build_trilitenet(preset, device):
    base = os.path.join(BASELINES, "TriLiteNet")
    sys.path.insert(0, base)
    sys.path.insert(0, os.path.join(base, "lib"))
    from lib.config import cfg  # noqa: E402
    from lib.models import get_net  # noqa: E402

    cfg.defrost()
    cfg.config = preset
    cfg.freeze()
    return get_net(cfg).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    args = ap.parse_args()

    device = torch.device(args.device)
    H, W = args.input_size
    x = torch.randn(1, 3, H, W)

    builders = [
        ("B1 YOLOP(separate repo, checked later)", None, None),
        ("B2a TwinLiteNet", build_twinlitenet, None),
        ("B2b TwinLiteNetPlus-nano", lambda d: build_twinlitenetplus("nano", d), None),
        ("B2b TwinLiteNetPlus-small", lambda d: build_twinlitenetplus("small", d), None),
        ("B3 TriLiteNet-tiny", lambda d: build_trilitenet("tiny", d), None),
        ("B3 TriLiteNet-small", lambda d: build_trilitenet("small", d), None),
        ("B3 TriLiteNet-base", lambda d: build_trilitenet("base", d), None),
    ]

    print(f"device={device}  input={H}x{W}")
    for name, build, _ in builders:
        if build is None:
            print(f"{'='*60}\n{name}: skipped in this script\n")
            continue
        try:
            model = build(device)
            n = count_params(model)
            flops = flops_of(model, x, device)
            dt = time_forward(model, x, device)
            print(f"{'='*60}\n{name}\n  params={n/1e6:.3f}M  flops={flops/1e9:.3f}G  "
                  f"fwd_time={dt*1e3:.2f}ms")
            # also try a plain CPU-forward sanity on a tiny input
            model.cpu()
            with torch.no_grad():
                _ = model(torch.randn(1, 3, 64, 96))
            print("  CPU tiny forward: OK")
        except Exception as e:  # noqa: BLE001
            print(f"{'='*60}\n{name}\n  FAILED: {type(e).__name__}: {e}")
        sys.path = [p for p in sys.path if p not in (
            os.path.join(BASELINES, "TwinLiteNet"),
            os.path.join(BASELINES, "TwinLiteNetPlus"),
            os.path.join(BASELINES, "TriLiteNet"),
            os.path.join(BASELINES, "TriLiteNet", "lib"),
        )]


if __name__ == "__main__":
    main()
