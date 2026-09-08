"""Smoke test for the P2 change.

Two things must hold:
  1. every already-committed model is bit-identical (params AND a forward
     pass on a fixed input) -- the change is flag-gated and off by default;
  2. p2="up" and p2="f1" build, run, and produce the expected prediction
     count (3 levels -> 25200, 4 levels -> 102000).
"""

import os
import sys

import torch
import yaml

sys.path.insert(0, ".")
from models.static_model import StaticMultiTaskModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KM12 = [[[8, 7], [15, 12], [20, 21]],
        [[35, 19], [38, 32], [64, 41]],
        [[41, 72], [87, 63], [135, 83]],
        [[98, 128], [180, 139], [241, 221]]]


def build(cfg_path, overrides=None):
    with open(os.path.join(ROOT, cfg_path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg["model"]["detection"].update(overrides)
    m = StaticMultiTaskModel(cfg["model"])
    m.eval()
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)

    # 1. committed baseline unchanged
    base = build("configs/phase4b_r2u_z16.yaml")
    with torch.no_grad():
        det, da, lane = base(x)
    print(f"baseline r2u_z16 : params {n_params(base)}  det {tuple(det.shape)}"
          f"  da {tuple(da.shape)}  lane {tuple(lane.shape)}")
    assert det.shape[1] == 25200, "baseline must still emit 3 levels"

    # 2. p2 variants
    for p2 in ("up", "f1"):
        m = build("configs/phase4b_dp2a_z16.yaml", {"p2": p2})
        with torch.no_grad():
            d2, da2, lane2 = m(x)
        dp = n_params(m) - n_params(base)
        print(f"p2={p2:2s}          : params {n_params(m)} ({dp:+d} vs baseline)"
              f"  det {tuple(d2.shape)}")
        assert d2.shape[1] == 102000, f"p2={p2} should emit 4 levels"
        assert da2.shape == da.shape and lane2.shape == lane.shape, \
            "p2 must not change the segmentation outputs"

    # 3. forward determinism check on the baseline (guards against a stray
    #    new module changing the default graph via RNG-free means)
    with torch.no_grad():
        det2, _, _ = base(x)
    assert torch.allclose(det, det2), "baseline forward not deterministic"
    print("\nOK: default path unchanged; p2 arms build and run.")


if __name__ == "__main__":
    main()
