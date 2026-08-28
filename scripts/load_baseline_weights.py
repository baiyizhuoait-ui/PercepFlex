#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline weight loader with compatibility handling.

- TwinLiteNet `pretrained/best.pth`: saved from DataParallel (`module.` prefix);
  we strip the prefix and load strict. Structure matches the current repo code.
- All other baselines load strict.

Usage:
    from scripts.load_baseline_weights import load_baseline
    model = load_baseline("TwinLiteNet", device)
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.join(ROOT, "baselines")
WEIGHTS = os.path.join(ROOT, "..", "trac_data", "weights")


def _strip_module(sd):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def load_twinlitenet(device="cpu"):
    sys.path.insert(0, os.path.join(BASELINES, "TwinLiteNet"))
    from model.TwinLite import TwinLiteNet  # noqa: E402

    model = TwinLiteNet().to(device)
    sd = torch.load(os.path.join(BASELINES, "TwinLiteNet", "pretrained", "best.pth"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = _strip_module(sd)  # saved from DataParallel; structure otherwise matches current code
    model.load_state_dict(sd, strict=True)
    return model


def load_twinlitenetplus(preset, device="cpu"):
    sys.path.insert(0, os.path.join(BASELINES, "TwinLiteNetPlus"))
    from model.model import TwinLiteNetPlus  # noqa: E402
    import argparse

    model = TwinLiteNetPlus(argparse.Namespace(config=preset)).to(device)
    sd = torch.load(os.path.join(WEIGHTS, "twinlitenetplus", f"{preset}.pth"),
                    map_location="cpu", weights_only=False)
    sd = _strip_module(sd.get("state_dict", sd))
    model.load_state_dict(sd, strict=True)
    return model


def load_trilitenet(preset, device="cpu"):
    """preset: 'tiny'|'small'|'base'; weight file nano->tiny, small, base."""
    base = os.path.join(BASELINES, "TriLiteNet")
    sys.path.insert(0, base)
    sys.path.insert(0, os.path.join(base, "lib"))
    from lib.config import cfg  # noqa: E402
    from lib.models import get_net  # noqa: E402

    cfg.defrost()
    cfg.config = preset
    cfg.freeze()
    model = get_net(cfg).to(device)
    wfile = {"tiny": "nano", "small": "small", "base": "base"}[preset]
    sd = torch.load(os.path.join(WEIGHTS, "trilitenet", f"{wfile}.pth"),
                    map_location="cpu", weights_only=False)
    sd = _strip_module(sd.get("state_dict", sd))
    sd = _strip_module(sd.get("model", sd))
    model.load_state_dict(sd, strict=True)
    return model


def load_baseline(name, preset=None, device="cpu"):
    if name == "TwinLiteNet":
        return load_twinlitenet(device)
    if name == "TwinLiteNetPlus":
        return load_twinlitenetplus(preset or "nano", device)
    if name == "TriLiteNet":
        return load_trilitenet(preset or "tiny", device)
    raise ValueError(f"unknown baseline {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    for name, presets in [("TwinLiteNet", [None]),
                          ("TwinLiteNetPlus", ["nano", "small", "medium", "large"]),
                          ("TriLiteNet", ["tiny", "small", "base"])]:
        for p in presets:
            m = load_baseline(name, p, args.device)
            print(f"{name} {p or ''}: OK params={sum(x.numel() for x in m.parameters())/1e6:.3f}M")
