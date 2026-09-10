"""Phase 6B / EXP-08 zero-training probe: FLOPs & params of uniform
encoder width expansion candidates. Target ~= 1.6391G (combo budget).
Builds StaticMultiTaskModel from scaled phase4a_r2_z16 configs and measures
with the same count_flops used by evaluation (no training, no GPU chain)."""
import sys

import torch
import yaml

sys.path.insert(0, ".")
from models.static_model import StaticMultiTaskModel  # noqa: E402
from profiling.flops_real import count_flops  # noqa: E402

TARGET_G = 1.6391
BASE = yaml.safe_load(open("configs/phase4a_r2_z16.yaml"))["model"]
x = torch.randn(1, 3, 640, 640)

print(f"target = {TARGET_G:.4f}G (combo budget)")
best = None
for s in [1.0, 1.2, 1.25, 1.3, 1.35]:
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    cfg["encoder"]["stem"] = max(8, int(round(BASE["encoder"]["stem"] * s / 8)) * 8)
    cfg["encoder"]["stages"] = [max(8, int(round(c * s / 8)) * 8)
                                for c in BASE["encoder"]["stages"]]
    m = StaticMultiTaskModel(cfg).eval()
    with torch.no_grad():
        f = count_flops(m, x)
    p = sum(pp.numel() for pp in m.parameters())
    gap = (f / 1e9 - TARGET_G) / TARGET_G * 100
    print(f"s={s:.2f} stem={cfg['encoder']['stem']} stages={cfg['encoder']['stages']} "
          f"params={p / 1e6:.4f}M flops={f / 1e9:.4f}G gap={gap:+.1f}%")
    if best is None or abs(f / 1e9 - TARGET_G) < abs(best[0] - TARGET_G):
        best = (f / 1e9, s, cfg["encoder"]["stem"], cfg["encoder"]["stages"], p)
print(f"BEST: s={best[1]:.2f} stem={best[2]} stages={best[3]} "
      f"params={best[4] / 1e6:.4f}M flops={best[0]:.4f}G")
