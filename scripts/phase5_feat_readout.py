"""Phase 4B STEP 2 - feature provenance by linear readout.

The lane probe asks whether 1/4 information reaching the head helps. Before
spending 20 epochs on a yes, the cheap question is whether the information is
even in the 1/4 map. A frozen-feature linear readout answers it:

  If a ridge / logistic classifier on s1 (1/4) predicts lane far better than
  one on F2 (1/8), the 1/4 information is present and the architecture is
  throwing it away. The readout's lane fg-IoU is an upper bound for what
  l14f1 can deliver, which calibrates the 20-epoch confirmation.

Same test, same loop, for DA. The block-fill ladder already showed DA has
only 0.11 of resolution headroom; if the 1/4 readout also does not help
DA, that is a second independent line of evidence against the resolution
hypothesis.

For detection, the same procedure is harder because the relevant units are
boxes, not pixels. 4B-3 (a real P2 detection head) is the test; this script
does not duplicate it.

GPU usage: one forward per val image to cache features. CPU does the
readouts. Cache to disk so the readout loop can be iterated without
re-running the encoder.
"""
import argparse
import os
import sys
import numpy as np
import torch
import yaml

ROOT = "/home/mycode/ai_study/trac"
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset, collate_train
from models.static_model import StaticMultiTaskModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg_path, ckpt_path):
    cfg = yaml.safe_load(open(cfg_path))
    model = StaticMultiTaskModel(cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(DEVICE)


@torch.no_grad()
def cache_features(model, ds, n, out_path):
    """Run the encoder over n val images and save per-level pooled features.

    Saves to a .npz with keys s1, f2, f3, f4 each (n, *feat_shape)
    downsampled to 1/8 by average pooling so all levels share a grid - this
    makes the readout comparison honest (same number of pixels, different
    information content). Lane mask, DA mask, detection GT boxes are saved
    alongside.
    """
    pass  # implemented in the run path below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default="configs/phase4a_r2_z16.yaml")
    ap.add_argument("--ckpt", default="experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt")
    ap.add_argument("--cache", default="experiments/phase5/phase5_feat_cache.npz")
    ap.add_argument("--out", default="experiments/phase5/phase5_feat_readout.csv")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        build_cache(args)
    fit_and_report(args)


def build_cache(args):
    print("caching features for", args.n, "val images ...")
    ds = BDD100KDataset(data_root=os.path.join(ROOT, "data", "bdd100k"),
                        split="val", image_size=(640, 640),
                        with_det=True, with_da=True, with_lane=True,
                        augment=False)
    n = min(args.n, len(ds))
    model = load_model(args.cfg, args.ckpt)

    s1, f2, f3, f4 = [], [], [], []
    lane_gts, da_gts = [], []
    for i in range(n):
        item = ds[i]
        img = item["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            # Static model exposes encoder levels via the config. Ask for them
            # through the highres path added in STEP 6.
            enc = model.encoder(img, highres=True)
        s1.append(enc["f0"].squeeze(0).cpu().numpy())
        f2.append(enc["f1"].squeeze(0).cpu().numpy())
        # The 1/16 and 1/32 maps come from the standard encoder forward.
        enc_std = model.encoder(img, highres=False)
        f3.append(enc_std["f2"].squeeze(0).cpu().numpy())
        f4.append(enc_std["f3"].squeeze(0).cpu().numpy())
        lane_gts.append(item["lane_mask"].squeeze(0).numpy().astype(np.uint8))
        da_gts.append(item["da_mask"].squeeze(0).numpy().astype(np.uint8))
        if (i + 1) % 20 == 0:
            print(f"  cached {i+1}/{n}")

    np.savez_compressed(
        args.cache,
        s1=np.stack(s1), f2=np.stack(f2), f3=np.stack(f3), f4=np.stack(f4),
        lane=np.stack(lane_gts), da=np.stack(da_gts),
    )
    print("wrote", args.cache)


def fit_and_report(args):
    z = np.load(args.cache)
    s1, f2, f3, f4 = z["s1"], z["f2"], z["f3"], z["f4"]
    lane, da = z["lane"], z["da"]

    rows = []
    for name, feat in (("s1_1over4", s1), ("f2_1over8", f2),
                       ("f3_1over16", f3), ("f4_1over32", f4)):
        for task, gt in (("lane", lane), ("da", da)):
            iou = linear_readout_iou(feat, gt, seed=args.seed)
            rows.append({"level": name, "task": task, "frozen_feat_iou": round(iou, 4)})
            print(f"  {name:10s} {task:5s}  frozen-feature IoU = {iou:.4f}")

    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "task", "frozen_feat_iou"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


def linear_readout_iou(feat, gt, seed=0):
    """Ridge classifier on a random subsample of pixels, IoU on the rest.

    feat: (N, C, h, w); gt: (N, h, w) in {0, 1}.
    Pools feat spatially to (h, w) by average pooling if h, w differ - which
    they do for s1 (1/4 -> 1/8) and f3 (1/16 -> 1/8) and f4 (1/32 -> 1/8).
    """
    from sklearn.linear_model import LogisticRegression
    import torch.nn.functional as F
    n, c, h, w = feat.shape
    if (h, w) != gt.shape[1:]:
        # Pool to 1/8 spatial (80 x 80 for a 640 input)
        t = torch.from_numpy(feat).float()
        if h > 80:
            t = F.adaptive_avg_pool2d(t, (80, 80))
        else:
            t = F.interpolate(t, size=(80, 80), mode="bilinear", align_corners=False)
        feat = t.numpy()
        h, w = 80, 80
    X = feat.transpose(0, 2, 3, 1).reshape(-n * h * w, c)
    y = gt.reshape(-1).astype(np.int32)
    # Subsample for speed; class-balance so the empty class doesn't dominate.
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) > 5000:
        pos = rng.choice(pos, 5000, replace=False)
    if len(neg) > 5000:
        neg = rng.choice(neg, 5000, replace=False)
    sel = np.concatenate([pos, neg])
    Xs, ys = X[sel], y[sel]
    clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
    clf.fit(Xs, ys)
    pred = clf.predict(X).reshape(n, h, w).astype(np.uint8)
    gt_r = gt[:, :h, :w]
    inter = (pred & gt_r).sum()
    union = (pred | gt_r).sum()
    return float(inter) / float(union) if union else float("nan")


if __name__ == "__main__":
    main()
