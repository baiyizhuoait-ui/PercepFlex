"""Phase 5 - error geometry, recomputed on the CONTENT region.

Why the Level 2 version had to be redone
----------------------------------------
phase4c_errorgeom.py measured lane and DA geometry on the full 640x640 canvas.
BDD100K frames are 1280x720 and are letterboxed to 640x640 with pad=(0,140),
so rows 0..139 and 500..639 are grey filler. Measured directly, the ground
truth has 0.0% of its drivable area and 0.0% of its lane area above row 213,
so a large part of the "image" being scored was padding the model is free to
paint. Any prediction the model makes in the filler inflates the false-positive
count and depresses IoU, and the vertical banding that produced the far-field
finding was measuring exactly that.

This version crops to the content region first, matching the official
evaluation, and recomputes:
  - fg IoU, precision, recall, predicted/gt area ratio
  - the 1/8 BLOCK-FILL reference (named honestly: the score of the strategy
    "paint every 1/8 cell the target touches", not a strict upper bound)
  - the boundary-tolerance curve at slack k = 0, 1, 2, 4, 8 px, which
    separates localisation error from missing-structure error
  - erosion thickness profile
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

from datasets.bdd100k import BDD100KDataset, collate                  # noqa: E402
from models.static_model import StaticMultiTaskModel                  # noqa: E402

OUT = os.environ.get("PHASE5_EG_OUT"
                     ) or os.path.join(ROOT, "experiments", "phase5",
                                       "phase5_error_geometry.csv")
TOLS = [0, 1, 2, 4, 8]
CELLS = [
    ("r0", "r0_z16",  "configs/phase3a_ebase_z16.yaml",
     "experiments/phase3a/exp3A_ebase_z16/checkpoint.pt"),
    ("r2", "r2_z16",  "configs/phase4a_r2_z16.yaml",
     "experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt"),
    ("r2", "r2_z32",  "configs/phase4a_r2_z32.yaml",
     "experiments/phase4a/exp4A_r2_z32_e20/checkpoint.pt"),
    ("r2", "r2_z128", "configs/phase4a_r2_z128.yaml",
     "experiments/phase4a/exp4A_r2_z128_e20/checkpoint.pt"),
]

# Phase 5 STEP 7: the lane probe cells are scored through this same code
# path. Set PHASE5_EG_CELLS to a JSON list of [variant, cell, config,
# checkpoint] rows to append them, so the probe is measured with the
# identical geometry code as the Phase 4A cells.
_extra = os.environ.get("PHASE5_EG_CELLS")
if _extra:
    import json as _json
    CELLS = CELLS + [tuple(x) for x in _json.loads(_extra)]

RES = 8


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def dilate(m, k):
    return m if k <= 0 else torch.nn.functional.max_pool2d(m, 2 * k + 1, 1, k)


def erode(m, k):
    return m if k <= 0 else -torch.nn.functional.max_pool2d(-m, 2 * k + 1, 1, k)


def fg_iou(pred, gt):
    inter = (pred & gt).float().sum()
    union = pred.float().sum() + gt.float().sum() - inter
    return (inter / union).item() if union > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=1)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for variant, cell, cfg_path, ckpt in CELLS:
        ck = os.path.join(ROOT, ckpt)
        if not os.path.exists(ck):
            print("[skip] %s" % cell)
            continue
        cfg = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
        model = build(cfg).to(dev).eval()
        model.load_state_dict(torch.load(ck, map_location="cpu")["model_state"],
                              strict=False)
        ds = BDD100KDataset(cfg["data"]["root"], split="tri_val")
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.bs, shuffle=False, num_workers=2,
            collate_fn=collate, drop_last=True)

        acc = {}
        for t in ("lane", "da"):
            acc[t] = {"iou": [], "prec": [], "rec": [], "area": [],
                      "blockfill": [], "gtpx": [],
                      "e1": [], "e2": []}
            for k in TOLS:
                acc[t]["iou%d" % k] = []
                acc[t]["rec%d" % k] = []
        seen = 0
        with torch.no_grad():
            for b in dl:
                if seen >= args.n:
                    break
                seen += b["image"].shape[0]
                out = model(b["image"].to(dev))
                _, da_lg, lane_lg = out[0], out[1], out[2]
                for i in range(b["image"].shape[0]):
                    ph = int(b["pad"][i][1])
                    nh = int(b["content_size"][i][0])
                    sl = slice(ph, ph + nh)
                    preds = {"lane": lane_lg[i:i + 1].argmax(1, keepdim=True).bool().cpu()[:, :, sl, :],
                             "da": da_lg[i:i + 1].argmax(1, keepdim=True).bool().cpu()[:, :, sl, :]}
                    gts = {"lane": (b["lane_mask"][i:i + 1, :, sl, :] > 0.5),
                           "da": (b["da_mask"][i:i + 1, :, sl, :] > 0.5)}
                    for t in ("lane", "da"):
                        g, p = gts[t], preds[t]
                        if g.sum().item() < 10:
                            continue
                        a = acc[t]
                        inter = (p & g).float().sum().item()
                        psum = p.float().sum().item()
                        gsum = g.float().sum().item()
                        a["iou"].append(inter / max(1e-9, psum + gsum - inter))
                        a["prec"].append(inter / max(1, psum))
                        a["rec"].append(inter / max(1, gsum))
                        a["area"].append(psum / max(1, gsum))
                        a["gtpx"].append(gsum)
                        a["e1"].append((erode(g.float(), 1) > 0.5).sum().item() / gsum)
                        a["e2"].append((erode(g.float(), 2) > 0.5).sum().item() / gsum)
                        up = torch.nn.functional.interpolate(
                            torch.nn.functional.max_pool2d(g.float(), RES, RES),
                            size=g.shape[-2:], mode="nearest") > 0.5
                        a["blockfill"].append(fg_iou(up, g))
                        for k in TOLS:
                            a["iou%d" % k].append(fg_iou(p, dilate(g.float(), k) > 0.5))
                            a["rec%d" % k].append(
                                (g & (dilate(p.float(), k) > 0.5)).sum().item() / gsum)

        def m(v):
            v = [x for x in v if x == x]
            return float(np.mean(v)) if v else float("nan")

        for t in ("lane", "da"):
            a = acc[t]
            row = {"cell": cell, "variant": variant, "task": t,
                   "fg_iou": round(m(a["iou"]), 4),
                   "precision": round(m(a["prec"]), 4),
                   "recall": round(m(a["rec"]), 4),
                   "pred_over_gt_area": round(m(a["area"]), 4),
                   "blockfill_1over8_iou": round(m(a["blockfill"]), 4),
                   "iou_minus_blockfill": round(m(a["iou"]) - m(a["blockfill"]), 4),
                   "erode1_survival": round(m(a["e1"]), 4),
                   "erode2_survival": round(m(a["e2"]), 4),
                   "mean_gt_px": round(m(a["gtpx"]), 1),
                   "n_images": seen}
            for k in TOLS:
                row["fg_iou_tol%d" % k] = round(m(a["iou%d" % k]), 4)
                row["recall_tol%d" % k] = round(m(a["rec%d" % k]), 4)
            rows.append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nError geometry on the CONTENT region (letterbox padding removed)")
    print("%-8s %-5s %8s %9s %9s %8s %10s %9s" %
          ("cell", "task", "fg_iou", "blockfill", "delta", "prec", "area", "recall"))
    for r in rows:
        print("%-8s %-5s %8.4f %9.4f %9.4f %8.4f %8.2f %8.4f" %
              (r["cell"], r["task"], r["fg_iou"], r["blockfill_1over8_iou"],
               r["iou_minus_blockfill"], r["precision"],
               r["pred_over_gt_area"], r["recall"]))
    print("\nBoundary tolerance curve (fg IoU at slack k px):")
    print("%-8s %-5s %s" % ("cell", "task", "".join("%9s" % ("k=%d" % k) for k in TOLS)))
    for r in rows:
        print("%-8s %-5s %s" % (r["cell"], r["task"],
                                "".join("%9.4f" % r["fg_iou_tol%d" % k] for k in TOLS)))
    print("\nWROTE %s" % OUT)


if __name__ == "__main__":
    main()
