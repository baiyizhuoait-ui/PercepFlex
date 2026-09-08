"""k-means (IoU distance) anchor clustering on BDD100K tri_train boxes.

Standard YOLOv5 auto-anchor procedure: distance = 1 - IoU(box, centroid),
boxes wh in input pixels on the 640x640 letterboxed canvas.

Emits K anchors sorted by area, ready to paste into a config's train.anchors.
Also reports the zero-match rate of the clustered set under the project's
assignment rule (0.5 < box/anchor < 2.0 on both axes), so we can see whether
clustering actually closes the coverage hole before spending GPU time.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from datasets.bdd100k import BDD100KDataset  # noqa: E402


def iou_wh(a, b):
    """a (n,2) b (m,2) -> (n,m) IoU of centre-aligned boxes."""
    inter = np.minimum(a[:, None, 0], b[None, :, 0]) * np.minimum(a[:, None, 1], b[None, :, 1])
    union = a[:, None, 0] * a[:, None, 1] + b[None, :, 0] * b[None, :, 1] - inter
    return inter / np.maximum(union, 1e-9)


def kmeans(boxes, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    n = boxes.shape[0]
    cent = boxes[rng.choice(n, k, replace=False)]
    for _ in range(iters):
        d = 1.0 - iou_wh(boxes, cent)
        assign = d.argmin(1)
        new = cent.copy()
        for j in range(k):
            m = assign == j
            if m.sum():
                new[j] = boxes[m].mean(0)
        if np.allclose(new, cent):
            break
        cent = new
    d = 1.0 - iou_wh(boxes, cent)
    return cent, d.min(1).mean(), assign


def zero_match(boxes, anchors, lo=0.5, hi=2.0):
    r = boxes[:, None, :] / anchors[None, :, :]
    ok = (r > lo).all(-1) & (r < hi).all(-1)
    return float((ok.sum(1) == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--n", type=int, default=3000, help="images to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = BDD100KDataset("data/bdd100k", split="tri_train")
    ws, hs = [], []
    for i in range(min(len(ds), args.n)):
        t = ds[i]["det_targets"]
        if t is None:
            continue
        t = np.asarray(t)
        if t.ndim != 2 or t.shape[0] == 0:
            continue
        ws.extend((t[:, 3] * 640.0).tolist())
        hs.extend((t[:, 4] * 640.0).tolist())
    boxes = np.stack([np.asarray(ws), np.asarray(hs)], 1)
    side = np.sqrt(boxes[:, 0] * boxes[:, 1])
    print(f"clustering {len(boxes)} boxes from {min(len(ds), args.n)} train images")
    print(f"  side: p10 {np.percentile(side,10):.1f} p50 {np.percentile(side,50):.1f} "
          f"p90 {np.percentile(side,90):.1f}")
    print(f"  aspect h/w median {np.median(boxes[:,1]/boxes[:,0]):.2f}")

    cent, avg_iou, _ = kmeans(boxes, args.k, seed=args.seed)
    order = np.argsort(cent[:, 0] * cent[:, 1])
    cent = cent[order]
    print(f"\nk={args.k} IoU-kmeans anchors (mean best IoU {1-avg_iou:.3f}):")
    for w, h in cent:
        print(f"  [{w:.0f}, {h:.0f}]   aspect {h/w:.2f}")

    cur = np.array([[4, 12], [7, 19], [11, 28], [17, 40], [25, 58], [38, 89],
                    [62, 136], [88, 206], [124, 412]], dtype=float)
    # IMPORTANT: report BOTH thresholds. The project's build_targets uses
    # 0.5 < box/anchor < 2.0. The YOLOv5 AutoAnchor standard (BPR) uses a
    # ratio threshold of 4.0 (0.25 < r < 4.0). A hole measured only under the
    # stricter rule is a fact about the ASSIGNMENT RULE as much as about the
    # anchors, and reporting it without the 4.0 column would be misleading.
    print("\nzero-match (no positive assignment) rate:")
    print(f"  {'anchor set':<22}{'thr=2.0 (project)':>20}{'thr=4.0 (YOLOv5 BPR)':>22}")
    for name, A in (("current 9", cur), ("k-means 9", cent)):
        z2 = zero_match(boxes, A, 0.5, 2.0)
        z4 = zero_match(boxes, A, 0.25, 4.0)
        print(f"  {name:<22}{100*z2:>19.1f}%{100*z4:>21.1f}%")
    sm = side < 32
    print("\n  small objects (side<32) only:")
    for name, A in (("current 9", cur), ("k-means 9", cent)):
        z2 = zero_match(boxes[sm], A, 0.5, 2.0)
        z4 = zero_match(boxes[sm], A, 0.25, 4.0)
        print(f"  {name:<22}{100*z2:>19.1f}%{100*z4:>21.1f}%")

    print("\npaste into config train.anchors (3 per level, ascending):")
    rows = [[f"[{w:.0f}, {h:.0f}]" for w, h in cent[i:i + 3]] for i in range(0, args.k, 3)]
    for r in rows:
        print("  - [" + ", ".join(r) + "]")


if __name__ == "__main__":
    main()
