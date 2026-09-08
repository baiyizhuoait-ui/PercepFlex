"""Zero-training diagnostic: does the CURRENT anchor set cover the GT boxes?

losses/yolo_loss.py build_targets() keeps a (level, anchor) pair for a target
only when  0.5 < box_px / anchor_px < 2.0  on BOTH axes. The stride cancels in
that ratio, so assignment depends only on the anchor's PIXEL size, not on the
level's stride.

Consequence: if the nine default anchors leave a size hole, the boxes falling
in it receive NO positive assignment at any level - they are trained as pure
negatives and can never be recovered by better features or more epochs. That
would be a supervision-side bottleneck, categorically different from the
resolution/grid story, and it must be ruled out before attributing the
small-object recall gap to grid resolution.

This script measures coverage of the current 3-level set, and of a candidate
4-level set whose stride-4 anchors are taken from the measured small-object
percentiles.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from datasets.bdd100k import BDD100KDataset  # noqa: E402

N_IMAGES = 500
CUR = np.array([[4, 12], [7, 19], [11, 28],
                [17, 40], [25, 58], [38, 89],
                [62, 136], [88, 206], [124, 412]], dtype=float)
# stride-4 anchors taken from the measured small-object percentiles
P2 = np.array([[9, 7], [14, 12], [24, 20]], dtype=float)
CAND = np.vstack([P2, CUR])


def match_count(boxes, anchors, lo=0.5, hi=2.0):
    """boxes (n,2) px; anchors (m,2) px -> (n,) number of matching anchors."""
    r = boxes[:, None, :] / anchors[None, :, :]
    ok = (r > lo).all(-1) & (r < hi).all(-1)
    return ok.sum(1)


def main():
    ds = BDD100KDataset("data/bdd100k", split="tri_val")
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
    print(f"n GT boxes = {len(boxes)}  ({N_IMAGES} val images)")
    print(f"small (side<32) = {(side<32).sum()} ({100*(side<32).mean():.1f}%)")

    for name, A in (("current 3-level (9 anchors)", CUR),
                    ("candidate 4-level (12 anchors)", CAND)):
        n = match_count(boxes, A)
        print(f"\n{name}")
        print(f"  boxes with ZERO matching anchor: {int((n==0).sum())} "
              f"({100*(n==0).mean():.1f}%)")
        print(f"  mean matches per box: {n.mean():.2f}   median {np.median(n):.0f}")
        sm = side < 32
        if sm.sum():
            ns = n[sm]
            print(f"  SMALL only: zero-match {int((ns==0).sum())}/{int(sm.sum())} "
                  f"({100*(ns==0).mean():.1f}%)   mean matches {ns.mean():.2f}")
        # which percentiles fall in the hole
        z = side[n == 0]
        if len(z):
            print(f"  zero-match box sizes: median side {np.median(z):.1f}, "
                  f"p10 {np.percentile(z,10):.1f}, p90 {np.percentile(z,90):.1f}")

    # the specific hole: median small object vs the current anchors
    med = np.array([np.median(boxes[side < 32, 0]), np.median(boxes[side < 32, 1])])
    print(f"\nmedian small object = {med[0]:.1f} x {med[1]:.1f} px")
    print("  ratio to each current anchor (need 0.5<r<2.0 on both axes):")
    for a in CUR:
        r = med / a
        tag = "MATCH" if (0.5 < r[0] < 2.0 and 0.5 < r[1] < 2.0) else "no"
        print(f"    anchor [{a[0]:5.0f},{a[1]:5.0f}]  ratio "
              f"({r[0]:5.2f},{r[1]:5.2f})  {tag}")


if __name__ == "__main__":
    main()
