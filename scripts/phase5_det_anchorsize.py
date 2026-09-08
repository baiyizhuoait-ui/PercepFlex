"""Measure the BDD100K detection box size distribution to SET P2 anchors from
data instead of guessing.

Two questions matter for the 4B-3 probe:
  1. what side lengths (sqrt(w*h)) does the "small" regime actually cover,
  2. how many grid cells does a small box occupy at stride 8 vs stride 4.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from datasets.bdd100k import BDD100KDataset  # noqa: E402

N = 500


def main():
    ds = BDD100KDataset("data/bdd100k", split="tri_val")
    it0 = ds[0]
    print("item keys:", sorted(it0.keys()))
    ws, hs = [], []
    for i in range(min(len(ds), N)):
        it = ds[i]
        b = it.get("det_targets")
        if b is None:
            continue
        b = np.asarray(b)
        if b.ndim != 2 or b.shape[0] == 0:
            continue
        # det_targets are (nt, 5): cls, cx, cy, w, h -- normalised to the
        # 640x640 letterboxed input. Convert w,h to input pixels.
        wh = b[:, 3:5].astype(float)
        if wh.size and wh.max() <= 2.0:      # normalised -> input pixels
            wh = wh * 640.0
        ws.extend(wh[:, 0].tolist())
        hs.extend(wh[:, 1].tolist())
    ws = np.asarray(ws)
    hs = np.asarray(hs)
    side = np.sqrt(ws * hs)
    print(f"\nn boxes = {len(ws)}  (from {N} val images)")
    print(f"small (side < 32): {int((side < 32).sum())} = {100*(side < 32).mean():.1f}%")
    print("\npercentile   side    w      h")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{q:<3}      {np.percentile(side,q):6.1f} {np.percentile(ws,q):6.1f} "
              f"{np.percentile(hs,q):6.1f}")
    sm = side < 32
    if sm.sum():
        sw, sh = ws[sm], hs[sm]
        print(f"\nsmall-only (n={int(sm.sum())}): "
              f"w median {np.median(sw):.1f} p10 {np.percentile(sw,10):.1f} p90 {np.percentile(sw,90):.1f} | "
              f"h median {np.median(sh):.1f} p10 {np.percentile(sh,10):.1f} p90 {np.percentile(sh,90):.1f}")
        print(f"  aspect h/w median {np.median(sh/sw):.2f}")
        # grid occupancy
        for s in (8, 4):
            cw, ch = sw / s, sh / s
            print(f"  at stride {s}: mean cells spanned {np.mean(cw):.2f} x {np.mean(ch):.2f}"
                  f"   (frac with min dim < 1 cell: {np.mean(np.minimum(cw,ch) < 1):.1%})")
    print("\nsuggested P2 anchors (stride 4), median-ish of the small regime:")
    # three anchors spread over the small regime, keeping the tall-vehicle aspect
    for q in (20, 50, 80):
        w = float(np.percentile(ws[sm], q)) if sm.sum() else 8.0
        h = float(np.percentile(hs[sm], q)) if sm.sum() else 18.0
        print(f"  [{w:.0f}, {h:.0f}]")


if __name__ == "__main__":
    main()
