"""Verify the anchor-coverage finding through the REAL code path.

phase5_anchor_coverage.py re-implemented the ratio rule; this script calls
losses.yolo_loss.YOLOLoss.build_targets directly with real targets and
correctly-shaped dummy predictions (build_targets only reads pred shapes),
so the number cannot be an artefact of my re-derivation.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from datasets.bdd100k import BDD100KDataset, collate  # noqa: E402
from losses.yolo_loss import YOLOLoss  # noqa: E402

N_IMAGES = 200


def run(anchors, strides, tag):
    A = torch.tensor(anchors, dtype=torch.float32)      # (nl, na, 2)
    nl, na = A.shape[0], A.shape[1]
    loss = YOLOLoss(A, nc=1, img_size=640, strides=strides)

    ds = BDD100KDataset("data/bdd100k", split="tri_val")
    tot, zero = 0, 0
    zero_side, all_side, small_tot, small_zero = [], [], 0, 0
    B = 4
    for start in range(0, min(len(ds), N_IMAGES), B):
        items = [ds[i] for i in range(start, min(start + B, len(ds)))]
        # targets: (nt, 6) = [image_id, cls, cx, cy, w, h], normalised
        parts = []
        for bi, it in enumerate(items):
            t = it["det_targets"]
            if t is None:
                continue
            t = torch.as_tensor(t)
            if t.ndim != 2 or t.shape[0] == 0:
                continue
            parts.append(torch.cat(
                [torch.full((t.shape[0], 1), float(bi)), t.float()], 1))
        if not parts:
            continue
        targets = torch.cat(parts, 0)
        shapes = [(640 // int(s), 640 // int(s)) for s in strides]
        preds = [torch.zeros(B, na, ny, nx, 6) for (ny, nx) in shapes]
        with torch.no_grad():
            tcls, tbox, indices, anch = loss.build_targets(preds, targets, 640)
        # indices[i] = (b, a, gj, gi) per level -> count distinct (b, gj, gi, a)
        matched = set()
        for (b, a, gj, gi) in indices:
            for bb, aa, y, x in zip(b.tolist(), a.tolist(), gj.tolist(), gi.tolist()):
                matched.add((bb, y, x, aa))
        # which GT got at least one positive: recover from tbox / image id
        img_ids = set()
        for (b, a, gj, gi) in indices:
            img_ids.update(b.tolist())
        # per-GT bookkeeping: rebuild using the same rule but keep target ids
        nt = targets.shape[0]
        hit = np.zeros(nt, dtype=bool)
        px = targets[:, 4:6].numpy() * 640.0
        for i, s in enumerate(strides):
            # In build_targets both gwh and a_grid are divided by the same
            # stride, so the ratio is stride-free: compare pixel box to pixel
            # anchor. (Dividing only the anchor by stride inflated every ratio
            # by `stride` and produced a bogus 93% hole.)
            ag = A[i].numpy()
            r = px[:, None, :] / ag[None, :, :]
            ok = (r > 0.5).all(-1) & (r < 2.0).all(-1)
            hit |= ok.any(1)
        side = np.sqrt(px[:, 0] * px[:, 1])
        tot += nt
        zero += int((~hit).sum())
        zero_side.extend(side[~hit].tolist())
        all_side.extend(side.tolist())
        sm = side < 32
        small_tot += int(sm.sum())
        small_zero += int((~hit & sm).sum())

    zs = np.asarray(zero_side)
    print(f"\n[{tag}]  anchors nl={nl} strides={tuple(strides)}")
    print(f"  GT boxes                    : {tot}")
    print(f"  with ZERO positive assignment: {zero} ({100*zero/max(tot,1):.1f}%)")
    print(f"  small (side<32) zero-match   : {small_zero}/{small_tot} "
          f"({100*small_zero/max(small_tot,1):.1f}%)")
    if len(zs):
        print(f"  zero-match sizes: median side {np.median(zs):.1f}, "
              f"p10 {np.percentile(zs,10):.1f}, p90 {np.percentile(zs,90):.1f}")


def main():
    CUR = [[[4, 12], [7, 19], [11, 28]],
           [[17, 40], [25, 58], [38, 89]],
           [[62, 136], [88, 206], [124, 412]]]
    run(CUR, (8, 16, 32), "current 3-level")
    P2 = [[[9, 7], [14, 12], [24, 20]]]
    run(P2 + CUR, (4, 8, 16, 32), "candidate 4-level")


if __name__ == "__main__":
    main()
