"""EXP-08 uniform probe extension: s in {1.4, 1.45, 1.5}."""
import sys

import torch
import yaml

sys.path.insert(0, ".")
from models.static_model import StaticMultiTaskModel  # noqa: E402
from profiling.flops_real import count_flops  # noqa: E402

BASE = yaml.safe_load(open("configs/phase4a_r2_z16.yaml"))["model"]
x = torch.randn(1, 3, 640, 640)
for s in [1.4, 1.45, 1.5]:
    cfg = yaml.safe_load(yaml.safe_dump(BASE))
    cfg["encoder"]["stem"] = max(8, int(round(BASE["encoder"]["stem"] * s / 8)) * 8)
    cfg["encoder"]["stages"] = [max(8, int(round(c * s / 8)) * 8)
                                for c in BASE["encoder"]["stages"]]
    m = StaticMultiTaskModel(cfg).eval()
    with torch.no_grad():
        f = count_flops(m, x)
    p = sum(pp.numel() for pp in m.parameters())
    st = cfg["encoder"]["stages"]
    sm = cfg["encoder"]["stem"]
    print(f"s={s:.2f} stem={sm} stages={st} params={p / 1e6:.4f}M "
          f"flops={f / 1e9:.4f}G gap={(f / 1e9 - 1.6391) / 1.6391 * 100:+.1f}%")
