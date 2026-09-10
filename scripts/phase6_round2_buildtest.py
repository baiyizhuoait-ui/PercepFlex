"""Build smoke test for Phase 6B new configs (zero-training, no GPU chain):
phase6_e7_z32_km.yaml and phase6_e8_uniform_z16.yaml.
Builds each model, runs one forward, reports params/FLOPs."""
import sys

import torch
import yaml

sys.path.insert(0, ".")
from models.static_model import StaticMultiTaskModel  # noqa: E402
from profiling.flops_real import count_flops  # noqa: E402

x = torch.randn(1, 3, 640, 640)
for path in ["configs/phase6_e7_z32_km.yaml", "configs/phase6_e8_uniform_z16.yaml"]:
    cfg = yaml.safe_load(open(path))["model"]
    m = StaticMultiTaskModel(cfg).eval()
    with torch.no_grad():
        out = m(x)
        f = count_flops(m, x)
    p = sum(pp.numel() for pp in m.parameters())
    shapes = [tuple(o.shape) for o in (out if isinstance(out, (list, tuple)) else [out])]
    print(f"{path}: params={p / 1e6:.4f}M flops={f / 1e9:.4f}G outs={shapes}")
print("BUILD SMOKE OK")
