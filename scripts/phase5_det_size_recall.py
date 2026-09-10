"""P4B-EXP-03, decision rule D2: is the anchor gain a *small-object* gain?

Zero training. Loads one or more frozen checkpoints and reports, per GT size
bucket, the fraction of GT boxes matched by at least one prediction at
IoU >= 0.5 (recall@0.5), the per-bucket AP@0.5, and the prediction count.

The preregistration (docs/PHASE5_DET_ANCHOR_PREREGISTRATION.md, rule D2) says
the danc gain only counts as evidence for H-19 if it lands on small-object
recall. mAP50 alone cannot show that, so this script stratifies by size.

Everything that defines a "prediction" is copied from
evaluation/evaluate_baseline.py: same decode, same NMS (conf=0.001, iou=0.6),
same dataset and same GT conversion (norm xywh -> xyxy px on 640x640).

Buckets (primary) use the preregistered definition "side < 32 px":
    small  : max(w, h) < 32
    medium : 32 <= max(w, h) < 96
    large  : max(w, h) >= 96
COCO area buckets (32^2 / 96^2) are printed alongside as a cross-check, so the
conclusion cannot hinge on one arbitrary threshold.

Usage
-----
    gpu_env/bin/python scripts/phase5_det_size_recall.py \
        --ckpt experiments/phase4a/exp4A_r2_z16_e4/checkpoint.pt=r2u_z16 \
        --ckpt experiments/phase4a/exp4B_danc_z16_e4/checkpoint.pt=danc_z16 \
        --out experiments/phase5/phase5_det_size_recall.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from datasets.bdd100k import BDD100KDataset            # noqa: E402
from evaluation.nms import non_max_suppression          # noqa: E402
from evaluate_baseline import build_ours, preprocess    # noqa: E402

CONF, IOU = 0.001, 0.6
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None)
    area_a = area_a[..., 0] * area_a[..., 1]
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None)
    area_b = area_b[..., 0] * area_b[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def ap_from(recall, precision):
    """COCO-style 101-point AP from a precision-recall curve."""
    if len(recall) == 0:
        return 0.0
    xs = np.linspace(0, 1, 101)
    return float(np.sum(np.interp(xs, recall, precision)) / 101.0)


def bucket_recall(ious, cols, iou_thr=0.5):
    """Exact recall@iou_thr: fraction of GT in `cols` with at least one pred
    at IoU >= thr. Uses ALL predictions (no top-K cap)."""
    if len(cols) == 0:
        return 0.0
    if ious.shape[0] == 0:
        return 0.0
    return float((ious[:, cols].max(axis=0) >= iou_thr).mean())


def bucket_ap(ious, scores, cols, iou_thr=0.5, max_preds=300):
    """AP@iou_thr for one bucket, greedy in descending score order.

    `ious` is the FULL (n_pred, n_gt) matrix for the image and `cols` selects
    this bucket's GT columns, so the matrix is built once per image.

    Approximation, stated up front: only the top `max_preds` predictions by
    confidence enter the AP curve (YOLO-style, COCO uses 100). With
    conf=0.001 the model emits ~5900 boxes/image, most of them noise; capping
    costs a negligible amount of AP and keeps this tractable. Recall, which is
    the D2 quantity, is NOT capped.
    """
    n_gt = len(cols)
    if n_gt == 0 or ious.shape[0] == 0:
        return 0.0, 0
    order = np.argsort(-scores)[:max_preds]
    sub = ious[np.ix_(order, cols)]
    matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(order), dtype=bool)
    for k in range(len(order)):
        row = sub[k].copy()
        row[matched] = -1.0
        j = int(row.argmax())
        if row[j] >= iou_thr:
            matched[j] = True
            tp[k] = True
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(~tp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return ap_from(rec, prec), len(order)


def side_bucket(gt):
    s = np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1])
    return np.where(s < 32, 0, np.where(s < 96, 1, 2))


def area_bucket(gt):
    a = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    return np.where(a < 32 * 32, 0, np.where(a < 96 * 96, 1, 2))


@torch.no_grad()
def run(ckpt, tag, split, device, max_images=None):
    model, _ = build_ours("OursStatic", ckpt, device)
    model.eval()
    ds = BDD100KDataset(os.path.join(ROOT, "data", "bdd100k"), split=split,
                        with_det=True, with_da=False, with_lane=False)
    if max_images:
        ds.names = ds.names[:max_images]
        if getattr(ds, "_det_by_name", None):
            keep = set(ds.names)
            ds._det_by_name = {k: v for k, v in ds._det_by_name.items() if k in keep}
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    all_pred, all_gt = [], []
    for item in loader:
        x = preprocess(item["image"][0], "unit")
        det_logits, _, _ = model(x)
        dets = non_max_suppression(det_logits, conf_thres=CONF, iou_thres=IOU)[0]
        gt = item["det_targets"][0][:, 1:] * 640            # norm xywh -> px
        gt = torch.cat([gt[:, :2] - gt[:, 2:] / 2, gt[:, :2] + gt[:, 2:] / 2], 1) \
            if len(gt) else gt
        all_pred.append((dets[:, :4].cpu().numpy(), dets[:, 4].cpu().numpy())
                        if len(dets) else (np.zeros((0, 4), np.float32),
                                           np.zeros((0,), np.float32)))
        all_gt.append(gt.numpy())

    # one IoU matrix per image, reused by every bucket
    per_image = []
    for (p, s), g in zip(all_pred, all_gt):
        if len(g) == 0:
            continue
        per_image.append((iou_matrix(p, g), s, g))

    rows = []
    for bname, bfn in (("side", side_bucket), ("area", area_bucket)):
        for bi, bsize in enumerate(("small", "medium", "large")):
            n_gt = 0
            matched = 0.0
            ap_num, ap_den = 0.0, 0
            for ious, s, g in per_image:
                cols = np.where(bfn(g) == bi)[0]
                if len(cols) == 0:
                    continue
                n_gt += len(cols)
                matched += bucket_recall(ious, cols, 0.5) * len(cols)
                a, _ = bucket_ap(ious, s, cols, 0.5)
                ap_num += a * len(cols)
                ap_den += len(cols)
            ap = ap_num / ap_den if ap_den else 0.0
            rows.append({"cell": tag, "bucket_by": bname, "bucket": bsize,
                         "n_gt": n_gt, "recall@0.5": round(matched / max(n_gt, 1), 4),
                         "AP@0.5": round(ap, 4)})
    # unstratified recall, as a sanity anchor against det_n_gt in metrics.json
    tot_gt = sum(len(g) for _, _, g in per_image)
    tot_rec = sum(bucket_recall(ious, np.arange(len(g)), 0.5) * len(g)
                  for ious, _, g in per_image)
    rows.append({"cell": tag, "bucket_by": "all", "bucket": "all",
                 "n_gt": tot_gt, "recall@0.5": round(tot_rec / max(tot_gt, 1), 4),
                 "AP@0.5": -1})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="path=tag (repeatable)")
    ap.add_argument("--split", default="tri_val")
    ap.add_argument("--out", default="experiments/phase5/phase5_det_size_recall.csv")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    all_rows = []
    for spec in args.ckpt:
        path, _, tag = spec.partition("=")
        tag = tag or os.path.basename(os.path.dirname(path))
        print(f"[size-recall] {tag} <- {path}", flush=True)
        all_rows += run(path, tag, args.split, device, args.max_images)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'cell':<10}{'by':<7}{'bucket':<9}{'n_gt':>8}{'recall@.5':>12}{'AP@.5':>9}")
    for r in all_rows:
        print(f"{r['cell']:<10}{r['bucket_by']:<7}{r['bucket']:<9}"
              f"{r['n_gt']:>8}{r['recall@0.5']:>12.4f}{r['AP@0.5']:>9.4f}")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
