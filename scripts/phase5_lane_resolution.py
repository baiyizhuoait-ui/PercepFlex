"""Phase 5 STEP 1 - lane resolution ladder, measured without training.

Why this is the first experiment
--------------------------------
At Level 2 the three trained models reached lane fg IoU 0.1843 / 0.1882 /
0.1845 while a model-free 1/8 ceiling - take the ground truth, max-pool it by
8, upsample it back, measure IoU - was 0.1853. The models do not merely
approach that ceiling, they are statistically indistinguishable from a crude
block quantisation of their own labels. That is the single strongest
measurement in the project and it has to be converted from an observation into
a causal claim before anything else is worth doing.

What this script measures
-------------------------
The same ceiling at factor 1, 2, 4, 8, 16 and 32, for both lane and DA. This
is the ladder that says how much room exists at each resolution, before any
architecture is changed:

  hard ceiling  fg IoU(gt, upsample(maxpool(gt, f)))
                The best any model can do if its task head only sees a
                binary decision at 1/f. Optimistic: it assumes the model
                detects a lane whenever a lane touches the cell.

  soft ceiling  fg IoU(gt, upsample(avgpool(gt, f)) > t*) with t* chosen
                per image to maximise IoU. This is the best a model that
                emits soft probabilities at 1/f could do with an oracle
                threshold. It is an upper bound on the hard ceiling and
                the right number to compare a trained model against.

Also reported per task:
  thickness profile   erode(gt, k).sum()/gt.sum() for k = 1, 2, 3. For a
                      stroke of width w, erosion by k survives iff w > 2k,
                      so this is a direct read-out of how thick the targets
                      actually are.
  precision / recall / area ratio of the hard ceiling prediction.

The prediction this ladder is designed to separate
--------------------------------------------------
H5a (head operating resolution) and H5b (absence of 1/4 information) are
different claims and only one of them can be decided by training:

  H5a predicts LITTLE gain from running the lane head at 1/4 while still
  feeding it Z. Z carries one vector per 1/8 cell and no information about
  where inside the cell the line sits, so a learned upsampler can at best
  emit a canonical pattern per cell. The ceiling for that is close to the
  1/8 ceiling. A large gain here would mean the models are leaving
  interpolation quality on the table, not that they lack information.

  H5b predicts a LARGE gain from giving the lane branch the encoder's real
  1/4 feature, because that feature genuinely contains the sub-cell position
  that Z has discarded.

So the ladder below bounds H5b from above before a single epoch is run. If
the 1/4 ceiling is not substantially higher than the 1/8 ceiling, H5b is
already dead and no training is justified.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = "/home/mycode/ai_study/trac"
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset, collate_train            # noqa: E402

OUT = os.path.join(ROOT, "experiments", "phase5", "phase5_resolution.csv")
FACTORS = [1, 2, 4, 8, 16, 32]
EROSIONS = [1, 2, 3]
THRESH_GRID = np.linspace(0.02, 0.98, 49)


def fg_iou(pred, gt):
    inter = (pred & gt).float().sum()
    union = pred.float().sum() + gt.float().sum() - inter
    return (inter / union).item() if union > 0 else float("nan")


def erode(m, k):
    if k <= 0:
        return m
    return -torch.nn.functional.max_pool2d(-m, 2 * k + 1, 1, k)


def hard_ceiling(gt, f):
    """Best binary output at 1/f resolution. Returns (iou, prec, rec, area)."""
    h, w = gt.shape[-2:]
    if f > 1:
        down = torch.nn.functional.max_pool2d(gt.float(), f, f)
        up = torch.nn.functional.interpolate(down, size=(h, w), mode="nearest")
    else:
        up = gt.float()
    pred = up > 0.5
    inter = (pred & gt).float().sum().item()
    psum = pred.float().sum().item()
    gsum = gt.float().sum().item()
    iou = inter / (psum + gsum - inter) if (psum + gsum - inter) > 0 else float("nan")
    prec = inter / psum if psum > 0 else float("nan")
    rec = inter / gsum if gsum > 0 else float("nan")
    area = psum / gsum if gsum > 0 else float("nan")
    return iou, prec, rec, area


def soft_ceiling(gt, f, mode="nearest"):
    """Oracle-thresholded soft output at 1/f. Returns (iou, best_threshold).

    Two variants, because for thin structures they disagree and the
    disagreement is itself informative:

      mode="nearest"   the tight upper bound. Each 1/f cell keeps its lane
                       fraction p (from average pooling); an oracle threshold
                       then decides which whole cells to emit. This is what a
                       perfect model could reach if it emitted a 1/f mask.

      mode="bilinear"  the same, but upsampled the way the actual segmentation
                       head upsamples its logits. Bilinear spreads one active
                       cell over roughly twice its footprint, which for 1-3 px
                       targets costs precision. It is lower, and it is the
                       number our architecture is actually compared against.
    """
    h, w = gt.shape[-2:]
    if f > 1:
        down = torch.nn.functional.avg_pool2d(gt.float(), f, f)
        up = torch.nn.functional.interpolate(
            down, size=(h, w), mode=mode,
            align_corners=False if mode == "bilinear" else None)
    else:
        up = gt.float()
    flat = up.reshape(-1)
    gsum = gt.float().sum().item()
    if gsum <= 0:
        return float("nan"), float("nan")
    best_i, best_t = -1.0, float("nan")
    # only thresholds that actually change the mask are worth testing
    cands = np.unique(np.quantile(flat.cpu().numpy(), THRESH_GRID))
    for t in cands:
        pred = up > float(t)
        inter = (pred & gt).float().sum().item()
        psum = pred.float().sum().item()
        den = psum + gsum - inter
        if den <= 0:
            continue
        i = inter / den
        if i > best_i:
            best_i, best_t = i, float(t)
    return best_i, best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "bdd100k"))
    args = ap.parse_args()

    ds = BDD100KDataset(args.root, split="tri_val")
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=2,
        collate_fn=collate_train, drop_last=True)

    acc = {}
    for task in ("lane", "da"):
        for f in FACTORS:
            acc[(task, f)] = {"iou": [], "prec": [], "rec": [], "area": [],
                              "soft": [], "thr": [], "bilin": []}
        acc[(task, "thick")] = {k: [] for k in EROSIONS}
        acc[(task, "px")] = []

    seen = 0
    with torch.no_grad():
        for b in dl:
            if seen >= args.n:
                break
            seen += b["image"].shape[0]
            masks = {"lane": b["lane_mask"] > 0.5, "da": b["da_mask"] > 0.5}
            for task, gt in masks.items():
                if not gt.any():
                    continue
                acc[(task, "px")].append(gt.float().sum().item())
                for k in EROSIONS:
                    acc[(task, "thick")][k].append(
                        (erode(gt.float(), k) > 0.5).sum().item()
                        / max(1, gt.sum().item()))
                for f in FACTORS:
                    i, p, r, a = hard_ceiling(gt, f)
                    sn, tn = soft_ceiling(gt, f, "nearest")
                    sb, _ = soft_ceiling(gt, f, "bilinear")
                    acc[(task, f)]["iou"].append(i)
                    acc[(task, f)]["prec"].append(p)
                    acc[(task, f)]["rec"].append(r)
                    acc[(task, f)]["area"].append(a)
                    acc[(task, f)]["soft"].append(sn)
                    acc[(task, f)]["thr"].append(tn)
                    acc[(task, f)]["bilin"].append(sb)

    def m(v):
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    rows = []
    for task in ("lane", "da"):
        base_h = m(acc[(task, 8)]["iou"])
        base_s = m(acc[(task, 8)]["soft"])
        for f in FACTORS:
            a = acc[(task, f)]
            h, w = 640 // f, 640 // f
            rows.append({
                "task": task,
                "factor": f,
                "out_res": "%dx%d" % (h, w),
                "hard_ceiling_fgiou": round(m(a["iou"]), 4),
                "soft_ceiling_fgiou": round(m(a["soft"]), 4),
                "bilinear_oracle_fgiou": round(m(a["bilin"]), 4),
                "oracle_threshold": round(m(a["thr"]), 3),
                "precision": round(m(a["prec"]), 4),
                "recall": round(m(a["rec"]), 4),
                "pred_over_gt_area": round(m(a["area"]), 4),
                "gain_vs_1over8_hard": round(m(a["iou"]) - base_h, 4),
                "gain_vs_1over8_soft": round(m(a["soft"]) - base_s, 4),
                "erode1_survival": round(m(acc[(task, "thick")][1]), 4),
                "erode2_survival": round(m(acc[(task, "thick")][2]), 4),
                "erode3_survival": round(m(acc[(task, "thick")][3]), 4),
                "mean_gt_px_per_img": round(m(acc[(task, "px")]), 1),
                "n_images": seen,
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    print("images measured: %d" % seen)
    print("")
    print("%-5s %5s %9s %11s %11s %11s %9s %8s" %
          ("task", "1/f", "out_res", "hard_ceil", "oracle_ceil",
           "bilinear", "gain_hard", "gain_ocl"))
    for r in rows:
        print("%-5s %5d %9s %11.4f %11.4f %11.4f %9.4f %8.4f" %
              (r["task"], r["factor"], r["out_res"],
               r["hard_ceiling_fgiou"], r["soft_ceiling_fgiou"],
               r["bilinear_oracle_fgiou"],
               r["gain_vs_1over8_hard"], r["gain_vs_1over8_soft"]))
    print("")
    print("thickness profile (erosion survival; stroke of width w survives k iff w > 2k):")
    for task in ("lane", "da"):
        r = next(x for x in rows if x["task"] == task and x["factor"] == 1)
        print("  %-5s erode1=%.4f  erode2=%.4f  erode3=%.4f  mean_gt_px=%.1f"
              % (task, r["erode1_survival"], r["erode2_survival"],
                 r["erode3_survival"], r["mean_gt_px_per_img"]))
    print("")
    print("WROTE %s" % OUT)


if __name__ == "__main__":
    main()
