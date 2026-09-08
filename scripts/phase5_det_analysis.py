"""Phase 5 STEP 3 - what kind of detection error is left, and where is it?

The question, stated precisely
------------------------------
Phase 3A/3C established that detection is the task that responds to encoder
capacity, and Phase 4A established that forcing detection through Z (R2) does
not hurt it - it helps. Neither says WHICH detection errors are being fixed.
"Encoder-sensitive" is not a bottleneck type. This script decomposes the
remaining error so it can be given a type.

Three decompositions, all zero-training
---------------------------------------
1. SIZE. COCO-style area buckets (small < 32^2, medium 32^2..96^2, large
   > 96^2 px) with per-bucket recall@0.5. H1 predicts the encoder advantage
   is concentrated on small objects. H2 predicts R2 (single 1/8 input) loses
   relative to R0 (multi-scale input) specifically on small objects.

2. POSITION. Recall by vertical position within the CONTENT region (letterbox
   padding removed - see the correction note in phase5_da_bands.py). Where
   errors sit vertically separates "far objects are too small" from "far
   objects are semantically hard".

3. CROWDING / OCCLUSION. For each ground-truth box, the maximum IoU it has
   with any other ground-truth box, and the number of ground truths in the
   image. H3 predicts recall degrades with crowding after controlling for
   size. If it does not, context is not the binding constraint and the
   dilation probe's failure was not a fluke of that particular design.

Matching is greedy one-to-one at IoU >= 0.5, the same convention the
project's own evaluate_detection uses, so these recalls are on the same
footing as the published mAP50.
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
from evaluation.nms import non_max_suppression                        # noqa: E402
from models.static_model import StaticMultiTaskModel                  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "phase5",
                   "phase5_detection_analysis.csv")

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
SMALL, MEDIUM = 32 * 32, 96 * 96


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def box_iou(b1, b2):
    if b1.numel() == 0 or b2.numel() == 0:
        return torch.zeros((b1.shape[0], b2.shape[0]))
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.001)
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
        sd = torch.load(ck, map_location="cpu")["model_state"]
        model.load_state_dict(sd, strict=False)

        ds = BDD100KDataset(cfg["data"]["root"], split="tri_val")
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.bs, shuffle=False, num_workers=2,
            collate_fn=collate, drop_last=True)

        rec = []       # (is_match, area, cy_rel, crowd_iou, n_gt_in_img)
        n_pred_tot = 0
        seen = 0
        with torch.no_grad():
            for batch in dl:
                if seen >= args.n:
                    break
                bs = batch["image"].shape[0]
                seen += bs
                x = batch["image"].to(dev)
                out = model(x)
                det = out[0] if isinstance(out, tuple) else out["det"]
                dets = non_max_suppression(det, conf_thres=args.conf,
                                           iou_thres=0.6)
                for i in range(bs):
                    t = batch["det_targets"][i]
                    if t.numel() == 0:
                        continue
                    xywh = t[:, 1:] * 640.0
                    gt = torch.stack([
                        xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                        xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
                    d = dets[i]
                    n_pred_tot += len(d)
                    pb = d[:, :4].cpu() if len(d) else torch.zeros((0, 4))
                    ious = box_iou(pb, gt) if len(pb) else torch.zeros((0, len(gt)))
                    best = ious.max(0).values if len(pb) else torch.zeros(len(gt))
                    # greedy one-to-one
                    matched = torch.zeros(len(gt), dtype=torch.bool)
                    if len(pb):
                        order = torch.argsort(-ious.max(1).values)
                        used = set()
                        for j in order.tolist():
                            k = int(ious[j].argmax())
                            if ious[j, k] >= 0.5 and k not in used:
                                used.add(k)
                                matched[k] = True
                    area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
                    pad_h = int(batch["pad"][i][1])
                    nh = int(batch["content_size"][i][0])
                    cy = (gt[:, 1] + gt[:, 3]) / 2
                    cy_rel = ((cy - pad_h) / max(1, nh)).clamp(0, 1)
                    gg = box_iou(gt, gt).fill_diagonal_(0)
                    crowd = gg.max(1).values
                    for m, a, c, cr in zip(matched.tolist(), area.tolist(),
                                           cy_rel.tolist(), crowd.tolist()):
                        rec.append((m, a, c, cr, len(gt)))

        if not rec:
            continue
        arr = np.array(rec, dtype=float)
        matched, area, cy, crowd, ngt = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

        def bucket_report(split, labels, order):
            for lb in order:
                sel = labels == lb if not isinstance(lb, tuple) else lb
                m = matched[sel]
                if len(m) == 0:
                    continue
                rows.append({
                    "cell": cell, "variant": variant, "split": split,
                    "bucket": str(lb), "n_gt": int(len(m)),
                    "recall_at50": round(float(m.mean()), 4),
                    "mean_gt_area_px": round(float(area[sel].mean()), 1),
                    "mean_crowd_iou": round(float(crowd[sel].mean()), 4),
                    "mean_gt_per_image": round(float(ngt[sel].mean()), 2),
                    "n_pred_total": n_pred_tot, "n_images": seen,
                })

        size_lab = np.where(area < SMALL, "small",
                            np.where(area < MEDIUM, "medium", "large"))
        bucket_report("size", size_lab, ["small", "medium", "large"])

        pos_lab = np.where(cy < 0.4, "upper", np.where(cy < 0.7, "middle", "lower"))
        bucket_report("position", pos_lab, ["upper", "middle", "lower"])

        cr_lab = np.where(crowd < 0.05, "isolated",
                          np.where(crowd < 0.30, "near", "overlapping"))
        bucket_report("crowding", cr_lab, ["isolated", "near", "overlapping"])

        dense_lab = np.where(ngt <= 5, "sparse",
                             np.where(ngt <= 15, "medium", "dense"))
        bucket_report("scene_density", dense_lab, ["sparse", "medium", "dense"])

        rows.append({
            "cell": cell, "variant": variant, "split": "overall",
            "bucket": "all", "n_gt": int(len(matched)),
            "recall_at50": round(float(matched.mean()), 4),
            "mean_gt_area_px": round(float(area.mean()), 1),
            "mean_crowd_iou": round(float(crowd.mean()), 4),
            "mean_gt_per_image": round(float(ngt.mean()), 2),
            "n_pred_total": n_pred_tot, "n_images": seen,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for split in ("size", "position", "crowding", "scene_density", "overall"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        cells = [c for _, c, _, _ in CELLS if c in {r["cell"] for r in sub}]
        buckets = []
        for r in sub:
            if r["bucket"] not in buckets:
                buckets.append(r["bucket"])
        print("\n--- %s (recall@0.5) ---" % split)
        print("%-12s %s" % ("bucket", "".join("%12s" % c for c in cells)))
        for b in buckets:
            line = "%-12s" % b
            for c in cells:
                v = [r["recall_at50"] for r in sub if r["cell"] == c and r["bucket"] == b]
                line += "%12.4f" % v[0] if v else "%12s" % "-"
            print(line)
        nb = "%-12s" % "n_gt"
        for c in cells:
            v = [r["n_gt"] for r in sub if r["cell"] == c and r["bucket"] == buckets[0]]
            nb += "%12d" % v[0] if v else "%12s" % "-"
        print(nb)
    print("\nWROTE %s" % OUT)


if __name__ == "__main__":
    main()
