#!/usr/bin/env python
"""P4A-STEP5 - LEVEL 0 diagnostics. No training is involved.

Two measurements, each attached to specific hypotheses:

A. Per-task gradient on Z   (H-01, H-02, H-05, H-10)

   ||dL_task / dZ|| for detection / DA / lane, plus the pairwise cosine between
   those three gradient vectors.

   In R0 the detection head never touches Z, so its gradient must be EXACTLY
   zero. In R2 it must be non-zero. That single number is the cleanest test
   available for the claim "the earlier Z results were masked by a bypass":
   zero means detection contributed no supervision to Z at all, non-zero means
   it did. The cosines then say whether the tasks pull Z in different
   directions (H-10).

B. Z utilisation via effective rank   (H-06, H-09, and sec.19)

   Per-channel activation statistics plus an SVD of the channel covariance.
   If z128 has roughly the same effective rank as z16, then the extra width is
   capacity the model does not actually use - which is a different conclusion
   from either "z128 helps" or "z128 is wasted", and is worth reporting on its
   own.

Both are computed from existing checkpoints only.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import yaml

ROOT = "/home/mycode/ai_study/trac"
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset, collate_train          # noqa: E402
from losses.multitask_loss import MultiTaskLoss, seg_ce_loss        # noqa: E402
from models.static_model import StaticMultiTaskModel                # noqa: E402

# Same default as training/train.py; the configs do not override it.
DEFAULT_ANCHORS = [[[4, 12], [7, 19], [11, 28]],
                   [[17, 40], [25, 58], [38, 89]],
                   [[62, 136], [88, 206], [124, 412]]]

OUT = os.path.join(ROOT, "experiments/phase4a")
GCSV = os.path.join(OUT, "phase4A_gradient_diagnostic.csv")
RCSV = os.path.join(OUT, "phase4A_effective_rank.csv")

CELLS = [
    ("r0", "r0_z16",  "configs/phase3a_ebase_z16.yaml",  "experiments/phase3a/exp3A_ebase_z16/checkpoint.pt"),
    ("r0", "r0_z32",  "configs/phase3b_ebase_z32.yaml",  "experiments/phase3b/exp3B_ebase_z32/checkpoint.pt"),
    ("r0", "r0_z128", "configs/phase3b_ebase_z128.yaml", "experiments/phase3b/exp3B_ebase_z128/checkpoint.pt"),
    ("r2", "r2_z16",  "configs/phase4a_r2_z16.yaml",  "experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt"),
    ("r2", "r2_z32",  "configs/phase4a_r2_z32.yaml",  "experiments/phase4a/exp4A_r2_z32_e20/checkpoint.pt"),
    ("r2", "r2_z128", "configs/phase4a_r2_z128.yaml", "experiments/phase4a/exp4A_r2_z128_e20/checkpoint.pt"),
]

TASKS = ["det", "da", "lane"]


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def load(cfg_path, ckpt, dev):
    cfg = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
    model = build(cfg).to(dev)
    sd = torch.load(os.path.join(ROOT, ckpt), map_location="cpu")["model_state"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("    [warn] missing=%s unexpected=%s" % (len(missing), len(unexpected)))
    return cfg, model


def batches_for(cfg, n, bs, dev, split):
    ds = BDD100KDataset(cfg["data"]["root"], split=split)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=2,
        collate_fn=collate_train, pin_memory=True, drop_last=True)
    out = []
    for i, b in enumerate(dl):
        if i >= n:
            break
        out.append((b["image"].to(dev, non_blocking=True),
                    b["det_targets"].to(dev),
                    b["da_mask"].to(dev),
                    b["lane_mask"].to(dev)))
    return out


def gradient_diagnostic(cfg, model, data, dev):
    """Per-task gradient norm wrt Z + pairwise cosine."""
    anchors = torch.tensor(cfg.get("train", {}).get("anchors", DEFAULT_ANCHORS),
                           dtype=torch.float32, device=dev)
    nc = cfg["model"].get("detection", {}).get("nc", 1)
    img_sz = cfg.get("input_size", 640)
    loss_fn = MultiTaskLoss(anchors, nc=nc, img_size=img_sz).to(dev)

    norms = {t: [] for t in TASKS}
    vecs = {t: [] for t in TASKS}
    model.train()

    for img, det_t, da_m, lane_m in data:
        model.zero_grad(set_to_none=True)
        det, da, lane, z = model(img, return_z=True)
        z.retain_grad()

        l_det, _ = loss_fn.det_loss(det, det_t, img_sz)
        l_da = seg_ce_loss(da, da_m)
        l_lane = seg_ce_loss(lane, lane_m)
        losses = {"det": l_det, "da": l_da, "lane": l_lane}

        for t in TASKS:
            model.zero_grad(set_to_none=True)
            z.grad = None
            losses[t].backward(retain_graph=True)
            g = z.grad
            if g is None:
                g = torch.zeros_like(z)
            g = g.detach().flatten().double()
            norms[t].append(g.norm().item())
            vecs[t].append(g)

    res = {"n_batches": len(data)}
    for t in TASKS:
        res["gnorm_" + t] = float(np.mean(norms[t]))
    tot = sum(res["gnorm_" + t] for t in TASKS)
    for t in TASKS:
        res["share_" + t] = res["gnorm_" + t] / tot if tot > 0 else float("nan")

    def cos(a, b):
        na, nb = a.norm(), b.norm()
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float((a @ b) / (na * nb))

    for x, y in [("det", "da"), ("det", "lane"), ("da", "lane")]:
        per = [cos(a, b) for a, b in zip(vecs[x], vecs[y])]
        per = [v for v in per if v == v]
        res["cos_%s_%s" % (x, y)] = float(np.mean(per)) if per else float("nan")
        mx = torch.stack(vecs[x]).mean(0)
        my = torch.stack(vecs[y]).mean(0)
        res["cosmean_%s_%s" % (x, y)] = cos(mx, my)
    return res


def rank_diagnostic(cfg, model, data, dev):
    """Effective rank and per-channel statistics of Z."""
    model.eval()
    C = None
    ssum = torch.zeros((1,), device=dev)
    cov = None
    act_list = []
    with torch.no_grad():
        for img, _, _, _ in data:
            _, _, _, z = model(img, return_z=True)
            B, c, h, w = z.shape
            C = c
            a = z.flatten(2)                       # (B, C, H*W)
            for i in range(B):
                act_list.append(a[i].double())     # (C, N)
    A = torch.stack(act_list)                      # (M, C, N)
    M, C, N = A.shape
    mu = A.mean(dim=(0, 2))                        # (C,)
    A = A - mu.view(1, C, 1)
    cov = torch.einsum("mcn,mkn->ck", A, A) / (M * N)
    cov = cov.float().cpu().numpy()
    s = np.linalg.svd(cov, compute_uv=False)
    s = np.clip(s, 1e-12, None)
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p)).sum()))
    cum = np.cumsum(p)[::-1] if False else np.cumsum(p)
    order = np.argsort(p)[::-1]
    pcum = np.cumsum(p[order])

    Aa = torch.stack(act_list).float()             # (M,C,N) uncentered copy for stats
    std = Aa.std(dim=(0, 2))
    mean_abs = Aa.abs().mean(dim=(0, 2))
    sparsity = float((Aa.abs() < 1e-3).float().mean().item())
    dead = int((std < 1e-6).sum().item())
    # mean absolute correlation between distinct channels (subsample for speed)
    flat = Aa.mean(dim=2).cpu().numpy()            # (M, C) channel means
    flat = flat - flat.mean(0, keepdims=True)
    nrm = np.linalg.norm(flat, axis=0, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    fn = flat / nrm
    corr = fn.T @ fn
    off = corr[~np.eye(C, dtype=bool)]
    return {
        "z_channels": C,
        "effective_rank": eff,
        "erank_frac_of_z": eff / C,
        "top1_var": float(pcum[0]),
        "top4_var": float(pcum[min(3, C - 1)]),
        "top8_var": float(pcum[min(7, C - 1)]),
        "top16_var": float(pcum[min(15, C - 1)]),
        "top32_var": float(pcum[min(31, C - 1)]),
        "mean_abs_corr": float(np.abs(off).mean()),
        "dead_channels": dead,
        "chan_std_mean": float(std.mean().item()),
        "chan_std_min": float(std.min().item()),
        "sparsity": sparsity,
        "act_abs_mean": float(mean_abs.mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="r0", choices=["r0", "r2", "all"])
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--split", default="tri_train")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    cells = [c for c in CELLS if args.variant == "all" or c[0] == args.variant]
    grad_rows, rank_rows = [], []
    for variant, cell, cfgp, ckp in cells:
        if not os.path.exists(os.path.join(ROOT, ckp)):
            print("  skip %s (no checkpoint)" % cell)
            continue
        print("  %s ..." % cell)
        cfg, model = load(cfgp, ckp, dev)
        data = batches_for(cfg, args.batches, args.bs, dev, args.split)
        g = gradient_diagnostic(cfg, model, data, dev)
        r = rank_diagnostic(cfg, model, data, dev)
        grad_rows.append(dict(variant=variant, cell=cell, **g))
        rank_rows.append(dict(variant=variant, cell=cell, **r))
        print("    gnorm det %.4f da %.4f lane %.4f | share_det %.3f | eff_rank %.2f / %d"
              % (g["gnorm_det"], g["gnorm_da"], g["gnorm_lane"],
                 g["share_det"], r["effective_rank"], r["z_channels"]))
        del model
        torch.cuda.empty_cache()

    def dump(path, rows):
        if not rows:
            print("  nothing to write for", path)
            return
        new = not os.path.exists(path)
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
        print("  written %s (%d rows)" % (path, len(rows)))

    dump(GCSV, grad_rows)
    dump(RCSV, rank_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
