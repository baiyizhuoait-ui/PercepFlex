"""Phase 6 EXP-3 (zero-training): size-stratified anchor/assignment analysis.

Question: in an extremely small model, is supervision quality (assignment)
amplified relative to representation quality? We quantify the assignment
improvement of the k-means anchor set over the old aspect-flipped default,
stratified by GT box size (COCO side buckets), and pair it with the measured
mAP effect sizes across runs to state the capacity-supervision coupling
hypothesis (H-M) with its current evidence tier.

Reuses datasets.bdd100k det_targets exactly like phase5_anchor_coverage.py.
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from datasets.bdd100k import BDD100KDataset  # noqa: E402

N_IMAGES = 800
OLD = np.array([[4, 12], [7, 19], [11, 28],
                [17, 40], [25, 58], [38, 89],
                [62, 136], [88, 206], [124, 412]], dtype=float)
KM = np.array([a for lvl in [[[9, 8], [18, 15], [32, 24]],
                             [[49, 38], [80, 52], [65, 102]],
                             [[124, 82], [166, 136], [237, 214]]] for a in lvl],
              dtype=float)


def best_iou(boxes, anchors):
    """boxes (n,2) wh px, anchors (m,2) wh px -> per-box best IoU + match count."""
    # intersection-over-union for axis-aligned boxes sharing a center
    w = np.minimum(boxes[:, None, 0], anchors[None, :, 0])
    h = np.minimum(boxes[:, None, 1], anchors[None, :, 1])
    inter = w * h
    union = (boxes[:, None, 0] * boxes[:, None, 1]) + \
            (anchors[None, :, 0] * anchors[None, :, 1]) - inter
    iou = inter / np.maximum(union, 1e-9)
    return iou.max(1)


def match_count(boxes, anchors, lo=0.5, hi=2.0):
    r = boxes[:, None, :] / anchors[None, :, :]
    ok = (r > lo).all(-1) & (r < hi).all(-1)
    return ok.sum(1)


def main():
    ds = BDD100KDataset("data/bdd100k", split="tri_train")
    bw, bh = [], []
    for i in range(min(len(ds), N_IMAGES)):
        t = ds[i]["det_targets"]
        if t is None:
            continue
        t = np.asarray(t)
        if t.ndim != 2 or t.shape[0] == 0:
            continue
        bw.extend((t[:, 3] * 640.0).tolist())
        bh.extend((t[:, 4] * 640.0).tolist())
    boxes = np.stack([np.asarray(bw), np.asarray(bh)], 1)
    side = np.sqrt(boxes[:, 0] * boxes[:, 1])
    buckets = {"small(<32)": side < 32,
               "medium(32-96)": (side >= 32) & (side < 96),
               "large(>=96)": side >= 96}

    lines = ["bucket,n,frac,old_mean_best_iou,km_mean_best_iou,d_iou,"
             "old_zeropos_frac,km_zeropos_frac"]
    print(f"n GT boxes = {len(boxes)} ({N_IMAGES} tri_train images)")
    lines.append(f"ALL,{len(boxes)},1.0,", )
    rows = []
    for name, m in [("ALL", np.ones_like(side, bool))] + list(buckets.items()):
        if m.sum() == 0:
            continue
        b = boxes[m]
        iou_o = best_iou(b, OLD)
        iou_k = best_iou(b, KM)
        mc_o = match_count(b, OLD)
        mc_k = match_count(b, KM)
        row = dict(bucket=name, n=int(m.sum()), frac=float(m.mean()),
                   old_iou=float(iou_o.mean()), km_iou=float(iou_k.mean()),
                   d_iou=float(iou_k.mean() - iou_o.mean()),
                   old_zp=float((mc_o == 0).mean()), km_zp=float((mc_k == 0).mean()))
        rows.append(row)
        print(f"{name:16s} n={row['n']:6d} ({100*row['frac']:5.1f}%)  "
              f"IoU {row['old_iou']:.3f} -> {row['km_iou']:.3f} "
              f"(+{row['d_iou']:.3f})  zeropos {100*row['old_zp']:5.1f}% -> "
              f"{100*row['km_zp']:4.1f}%")
        lines.append(f"{name},{row['n']},{row['frac']:.4f},{row['old_iou']:.4f},"
                     f"{row['km_iou']:.4f},{row['d_iou']:.4f},{row['old_zp']:.4f},"
                     f"{row['km_zp']:.4f}")
    with open("experiments/phase6/phase6_exp3_assignment.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE experiments/phase6/phase6_exp3_assignment.csv")


if __name__ == "__main__":
    main()
