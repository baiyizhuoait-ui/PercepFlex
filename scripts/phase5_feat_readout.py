"""P4B-STEP2 (P4B-EXP-02) - feature provenance by linear readout.

Question: does the trained BASELINE encoder's 1/4 map (f1, discarded by the
architecture) linearly encode lane location better than its 1/8 map (f2,
what the lane head actually reads)?

  If a logistic probe on f1 predicts lane cells far better than one on f2,
  the information exists and the architecture throws it away (H-05b is about
  the path, not the encoding). If f1 ~= f2, the encoder never encoded lane
  detail at 1/4 and the l14f1 gain must come from the lateral conv learning
  lane-specific features, not from pre-existing encoder information.

Protocol (literature: Alain & Bengio 2016 probing; DINOv2 seg probes use a
1x1 conv on frozen features): all feature maps are brought to the same
1/8 cell grid (avg-pool down, bilinear up) so the level comparison holds
the number of cells constant; GT cells are max-pooled to the same grid so
the target is identical for every level. IoU is measured at cell level -
absolute values are NOT the block-fill ceiling; only cross-level
comparisons are meaningful. Letterbox pad cells are all-negative GT and
affect every level equally.

Secondary row: f1 probed at its native 1/4 grid (160x160 cells) to check
whether pooling to 1/8 destroys the signal.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = "/home/mycode/ai_study/trac"
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset  # noqa: E402
from models.static_model import StaticMultiTaskModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg_path, ckpt_path):
    cfg = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    model = StaticMultiTaskModel(flat)
    ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu")
    model.load_state_dict(ck["model_state"], strict=False)
    return model.eval().to(DEVICE)


@torch.no_grad()
def build_cache(args):
    print("caching features for", args.n, "tri_val images ...")
    ds = BDD100KDataset(os.path.join(ROOT, "data", "bdd100k"),
                        split="tri_val", img_size=640,
                        with_det=False, with_da=True, with_lane=True,
                        train=False)
    model = build_model(args.cfg, args.ckpt)
    n = min(args.n, len(ds))
    f1s, f2s, f3s, f4s, lanes, das = [], [], [], [], [], []
    for i in range(n):
        item = ds[i]
        img = item["image"].unsqueeze(0).to(DEVICE)
        hi, lo = model.encoder(img, highres=True)
        f1s.append(lo["f1"].squeeze(0).cpu().numpy())   # 1/4
        f2s.append(hi[0].squeeze(0).cpu().numpy())      # 1/8
        f3s.append(hi[1].squeeze(0).cpu().numpy())      # 1/16
        f4s.append(hi[2].squeeze(0).cpu().numpy())      # 1/32
        lane = item["lane_mask"].squeeze(0).numpy()
        da = item["da_mask"].squeeze(0).numpy()
        lanes.append(((lane > 0) & (lane < 255)).astype(np.uint8))
        das.append((da > 0).astype(np.uint8))
        if (i + 1) % 25 == 0:
            print("  cached %d/%d" % (i + 1, n))
    np.savez_compressed(args.cache,
                        f1=np.stack(f1s), f2=np.stack(f2s),
                        f3=np.stack(f3s), f4=np.stack(f4s),
                        lane=np.stack(lanes), da=np.stack(das))
    print("wrote", args.cache)


def cell_iou_probe(feat, gt_cells, seed=0, train_cells=5000):
    """Logistic probe on per-cell features; IoU on ALL cells.

    feat: (N, C, h, w); gt_cells: (N, h, w) in {0,1}.
    """
    from sklearn.linear_model import LogisticRegression
    n, c, h, w = feat.shape
    X = feat.transpose(0, 2, 3, 1).reshape(n * h * w, c).astype(np.float32)
    y = gt_cells.reshape(-1).astype(np.int32)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if len(pos) > train_cells:
        pos = rng.choice(pos, train_cells, replace=False)
    if len(neg) > train_cells:
        neg = rng.choice(neg, train_cells, replace=False)
    sel = np.concatenate([pos, neg])
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(X[sel], y[sel])
    pred = clf.predict(X).reshape(n, h, w).astype(bool)
    g = gt_cells.astype(bool)
    inter = float((pred & g).sum())
    union = float((pred | g).sum())
    return inter / union if union else float("nan")


def to_cells(feat, grid):
    """Avg-pool down / bilinear-up a (N,C,h,w) feature map to grid x grid."""
    t = torch.from_numpy(feat).float()
    _, _, h, w = t.shape
    if h > grid:
        t = F.adaptive_avg_pool2d(t, (grid, grid))
    elif h < grid:
        t = F.interpolate(t, size=(grid, grid), mode="bilinear",
                          align_corners=False)
    return t.numpy()


def gt_to_cells(gt, grid):
    """Max-pool a (N,H,W) binary mask to grid x grid: cell = any pixel."""
    t = torch.from_numpy(gt).float().unsqueeze(1)
    k = gt.shape[-1] // grid
    t = F.max_pool2d(t, k)
    return t.squeeze(1).numpy()


def fit_and_report(args):
    z = np.load(args.cache)
    feats = {"enc_1over4": z["f1"], "enc_1over8": z["f2"],
             "enc_1over16": z["f3"], "enc_1over32": z["f4"]}
    rows = []
    for task in ("lane", "da"):
        gt = z[task]
        g80 = gt_to_cells(gt, 80)
        g160 = gt_to_cells(gt, 160)
        for name, feat in feats.items():
            iou = cell_iou_probe(to_cells(feat, 80), g80, seed=args.seed)
            rows.append({"level": name, "task": task,
                         "frozen_cell_iou": round(iou, 4)})
            print("  %-12s %-4s cell-IoU@1/8 = %.4f" % (name, task, iou))
        # native-resolution probe for the 1/4 map only
        iou = cell_iou_probe(to_cells(feats["enc_1over4"], 160), g160,
                             seed=args.seed)
        rows.append({"level": "enc_1over4_native", "task": task,
                     "frozen_cell_iou": round(iou, 4)})
        print("  %-12s %-4s cell-IoU@1/4 = %.4f" %
              ("enc_1over4_native", task, iou))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "task",
                                          "frozen_cell_iou"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="configs/phase4a_r2_z16.yaml")
    ap.add_argument("--ckpt",
                    default="experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt")
    ap.add_argument("--cache", default="experiments/phase5/phase5_feat_cache.npz")
    ap.add_argument("--out", default="experiments/phase5/phase5_feat_readout.csv")
    args = ap.parse_args()
    if not os.path.exists(args.cache):
        build_cache(args)
    fit_and_report(args)


if __name__ == "__main__":
    main()
