#!/usr/bin/env python
"""Level 2 - error geometry of the segmentation tasks. No training.

Question this answers
---------------------
Three probes (A per-task width, B reconstruction, C gradient balance) all failed
to move DA or lane by acting on the shared Z. The leading remaining explanation
is spatial, not capacity: every task reads Z at 1/8 resolution, and a lane
marking a few pixels wide is a sub-pixel feature there. This script tests that
without training anything.

Three measurements:

1. RESOLUTION CEILING (model-free). Take the ground-truth lane mask, max-pool it
   down by 8 (what a 1/8 feature map can retain at best), upsample it back, and
   measure IoU against the original. This is an upper bound on lane IoU for ANY
   model whose lane head only sees 1/8 features. If it is far below 1.0, the
   ceiling is geometric and no amount of Z capacity can lift it.

2. THICKNESS. erode(mask, 1).sum() / mask.sum(). Near 0 means the structure is
   thin (lines); near 1 means it is blob-like. Lane should be thin, DA blob-like.

3. BOUNDARY-TOLERANCE CURVE (model). IoU(pred, dilate(gt, k)) and recall@k =
   fraction of gt foreground within k pixels of a prediction, for k = 0,1,2,4,8.
   A steep curve means the errors are localisation/boundary errors - consistent
   with a resolution limit. A flat curve means the head is missing whole
   structures, which is a different failure and would point at supervision or
   loss weighting instead.

DA is additionally bucketed into top / middle / bottom thirds of the image
(far / mid / near) to see whether its residual errors are horizon cases.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

ROOT = "/home/mycode/ai_study/trac"
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset, collate_train          # noqa: E402
from models.static_model import StaticMultiTaskModel                # noqa: E402

OUT = os.path.join(ROOT, "experiments", "phase4a")

CELLS = [
    ("r2", "r2_z16",  "configs/phase4a_r2_z16.yaml",  "experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt"),
    ("r2", "r2_z128", "configs/phase4a_r2_z128.yaml", "experiments/phase4a/exp4A_r2_z128_e20/checkpoint.pt"),
    ("r0", "r0_z16",  "configs/phase3a_ebase_z16.yaml", "experiments/phase3a/exp3A_ebase_z16/checkpoint.pt"),
]

TOLS = [0, 1, 2, 4, 8]


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def dilate(m, k):
    if k <= 0:
        return m
    return torch.nn.functional.max_pool2d(m, 2 * k + 1, 1, k)


def erode(m, k):
    if k <= 0:
        return m
    return -torch.nn.functional.max_pool2d(-m, 2 * k + 1, 1, k)


def iou(pred, gt):
    inter = (pred & gt).float().sum()
    union = (pred | gt).float().sum()
    return (inter / union).item() if union > 0 else float("nan")


def fg_iou(pred, gt):
    """Foreground-only IoU, matching SegmentationMetric.fg_iou semantics."""
    inter = (pred & gt).float().sum()
    union = (pred.float().sum() + gt.float().sum() - inter)
    return (inter / union).item() if union > 0 else float("nan")


def res_ceiling(gt, factor=8):
    """IoU(gt, upsample(maxpool(gt, factor))) - best case for a 1/factor map."""
    n, _, h, w = gt.shape
    ph = max(1, h // factor)
    pw = max(1, w // factor)
    down = torch.nn.functional.max_pool2d(gt.float(), factor, factor)
    up = torch.nn.functional.interpolate(down, size=(h, w), mode="nearest")
    return fg_iou(up > 0.5, gt.bool()), (ph, pw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of val images")
    ap.add_argument("--bs", type=int, default=1)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    detail = []

    for variant, cell, cfg_path, ckpt in CELLS:
        ck = os.path.join(ROOT, ckpt)
        if not os.path.exists(ck):
            print("[skip] %s - no checkpoint at %s" % (cell, ckpt))
            continue
        cfg = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
        model = build(cfg).to(dev).eval()
        sd = torch.load(ck, map_location="cpu")["model_state"]
        model.load_state_dict(sd, strict=False)

        ds = BDD100KDataset(cfg["data"]["root"], split="val")
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.bs, shuffle=False, num_workers=2,
            collate_fn=collate_train, drop_last=True)

        acc = {"lane_ceil": [], "lane_erode": [], "da_ceil": [], "da_erode": [],
               "lane_gtpx": [], "lane_predpx": [], "da_h": [], "lane_h": []}
        lane_iou = {k: [] for k in TOLS}
        lane_rec = {k: [] for k in TOLS}
        da_iou = {k: [] for k in TOLS}
        da_band = {"top": [], "mid": [], "bot": []}

        seen = 0
        with torch.no_grad():
            for b in dl:
                if seen >= args.n:
                    break
                x = b["image"].to(dev)
                gt_da = b["da_mask"].to(dev)
                gt_lane = b["lane_mask"].to(dev)
                out = model(x)
                if isinstance(out, tuple) and len(out) == 3:
                    _, da_lg, lane_lg = out
                else:
                    det, da_lg, lane_lg = out[0], out[1], out[2]

                H, W = da_lg.shape[-2:]
                gda = torch.nn.functional.interpolate(
                    gt_da.float(), size=(H, W), mode="nearest")
                gln = torch.nn.functional.interpolate(
                    gt_lane.float(), size=(H, W), mode="nearest")
                gda_b = gda > 0.5
                gln_b = gln > 0.5
                pda_b = da_lg.argmax(1, keepdim=True).bool()
                pln_b = lane_lg.argmax(1, keepdim=True).bool()

                if gda_b.any():
                    c, _ = res_ceiling(gda_b, 8)
                    acc["da_ceil"].append(c)
                    acc["da_erode"].append(
                        (erode(gda_b.float(), 1) > 0.5).sum().item() / max(1, gda_b.sum().item()))
                if gln_b.any():
                    c, _ = res_ceiling(gln_b, 8)
                    acc["lane_ceil"].append(c)
                    acc["lane_erode"].append(
                        (erode(gln_b.float(), 1) > 0.5).sum().item() / max(1, gln_b.sum().item()))
                    acc["lane_gtpx"].append(gln_b.sum().item())
                    acc["lane_predpx"].append(pln_b.sum().item())

                for k in TOLS:
                    if gln_b.any():
                        lane_iou[k].append(fg_iou(pln_b, dilate(gln_b.float(), k) > 0.5))
                        lane_rec[k].append(
                            (gln_b & (dilate(pln_b.float(), k) > 0.5)).sum().item()
                            / max(1, gln_b.sum().item()))
                    if gda_b.any():
                        da_iou[k].append(fg_iou(pda_b, dilate(gda_b.float(), k) > 0.5))

                thirds = H // 3
                for name, sl in [("top", slice(0, thirds)),
                                 ("mid", slice(thirds, 2 * thirds)),
                                 ("bot", slice(2 * thirds, H))]:
                    g = gda_b[:, :, sl, :]
                    p = pda_b[:, :, sl, :]
                    if g.any():
                        da_band[name].append(fg_iou(p, g))

                seen += x.shape[0]

        def avg(v):
            return float(np.mean(v)) if v else float("nan")

        row = {
            "variant": variant, "cell": cell, "n_images": seen, "out_res": "%dx%d" % (H, W),
            "lane_res_ceiling_1over8": round(avg(acc["lane_ceil"]), 4),
            "lane_erode1_ratio": round(avg(acc["lane_erode"]), 4),
            "lane_gt_px_per_img": round(avg(acc["lane_gtpx"]), 1),
            "lane_pred_over_gt_area": round(
                avg(acc["lane_predpx"]) / max(1e-9, avg(acc["lane_gtpx"])), 4),
            "da_res_ceiling_1over8": round(avg(acc["da_ceil"]), 4),
            "da_erode1_ratio": round(avg(acc["da_erode"]), 4),
            "da_iou_top": round(avg(da_band["top"]), 4),
            "da_iou_mid": round(avg(da_band["mid"]), 4),
            "da_iou_bottom": round(avg(da_band["bot"]), 4),
        }
        for k in TOLS:
            row["lane_fgiou_tol%d" % k] = round(avg(lane_iou[k]), 4)
            row["lane_recall_tol%d" % k] = round(avg(lane_rec[k]), 4)
            row["da_fgiou_tol%d" % k] = round(avg(da_iou[k]), 4)
        rows.append(row)
        print("[done] %-8s n=%d out=%s lane_ceil(1/8)=%.4f lane_erode1=%.4f "
              "lane_fgiou@0=%.4f @8=%.4f" % (
                  cell, seen, row["out_res"], row["lane_res_ceiling_1over8"],
                  row["lane_erode1_ratio"], row["lane_fgiou_tol0"],
                  row["lane_fgiou_tol8"]))

    if rows:
        path = os.path.join(OUT, "phase4C_error_geometry.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n[written] %s" % path)

        print("\n--- boundary-tolerance curves (foreground IoU) ---")
        print("  %-8s %s" % ("cell", "  ".join("tol%-5d" % k for k in TOLS)))
        for r in rows:
            print("  %-8s %s" % (r["cell"], "  ".join(
                "%.4f " % r["lane_fgiou_tol%d" % k] for k in TOLS)))
        print("\n--- lane recall@k (fraction of gt lane pixels within k px of a pred) ---")
        for r in rows:
            print("  %-8s %s" % (r["cell"], "  ".join(
                "%.4f " % r["lane_recall_tol%d" % k] for k in TOLS)))


if __name__ == "__main__":
    main()
