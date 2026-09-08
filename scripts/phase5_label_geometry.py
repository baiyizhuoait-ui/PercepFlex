"""Phase 5 STEP 4 - is the lane number partly the label's fault, not the model's?

Why this exists
---------------
Lane foreground IoU is the worst number in the whole project (0.176-0.188) and
it barely moves. Two explanations have been on the table:

  H5   the head operates at 1/8 and cannot place a 3-4 px line - a capacity /
       resolution fact about the model.
  H8  the label itself is the problem: BDD100K lane masks are drawn with a
       stroke width, the train and val strokes are reported to differ
       (8 px train vs 2 px test is the figure quoted in the lane-detection
       literature), and a pixel-level IoU between two strokes of different
       width is bounded by min(w1,w2)/max(w1,w2) even when the model localises
       the line perfectly.

These are not the same claim and they do not have the same fix. If the second
one is large, then a chunk of the lane deficit was never available to any
architecture, and reading it as a modelling failure is wrong.

This script measures the stroke geometry directly. No model, no training.

What it measures
----------------
For lane, over N sampled images from each split:
  fg_share        foreground pixel share
  width_mean      4 * mean(EDT over foreground). For an ideal straight stroke
                  of width w the EDT rises linearly to w/2, so the mean over
                  the stroke is w/4. Curvature biases this down slightly.
  width_p99       2 * 99th percentile of EDT. The deepest pixel in a stroke is
                  w/2 from each edge, so this estimates w and is much less
                  curvature-sensitive than the mean. Preferred estimator.
  erode_k1/k2     share of foreground surviving erosion with a 3x3 / 5x5
                  kernel. Reproduces the earlier 0.0107 / 0.0007 as a sanity
                  check that we are looking at the same masks.
  boundary_share  share of foreground within 1 px of the background.

  iou_if_pred_w   the IoU a *perfectly localised* predictor would get if it
                  painted a stroke of width w_pred on the val centreline, for
                  w_pred in a small grid. For two concentric strokes this is
                  min/max, so it is a hard bound independent of any model.

For DA, same sample:
  fg_share, boundary_share, n_components, area.

Widths are reported at native resolution (1280x720) and at model-input scale
(x0.5, because 1280x720 letterboxes to 640x640 with pad 140 top and bottom).
The model-input number is the one that matters for comparing against the
3-4 px figure already in the notes.
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = "/home/mycode/ai_study/trac"
DATA = os.path.join(ROOT, "data", "bdd100k")
OUT_CSV = os.path.join(ROOT, "experiments", "phase5", "phase5_label_geometry.csv")
OUT_LOG = os.path.join(ROOT, "experiments", "phase5", "exp5_label_geometry.log")

INPUT_SCALE = 0.5  # 1280x720 -> 640x640 letterbox, content rows 140..500


def list_names(split):
    p = os.path.join(DATA, "splits", f"lane_{split}.txt")
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_mask(kind, split, name):
    p = os.path.join(DATA, kind, "masks", split, name + ".png")
    with Image.open(p) as im:
        return np.asarray(im)


def stroke_width(fg):
    """Return (width_mean, width_p99) in pixels for a binary mask."""
    if fg.sum() == 0:
        return float("nan"), float("nan")
    edt = ndimage.distance_transform_edt(fg)
    v = edt[fg]
    return float(4.0 * v.mean()), float(2.0 * np.percentile(v, 99))


def concentric_iou(fg, w_pred):
    """IoU of a stroke of width w_pred centred on fg's centreline, vs fg.

    Approximated by eroding/dilating: if w_pred > w, prediction is fg dilated
    by (w_pred - w)/2; if smaller, it is fg eroded by (w - w_pred)/2. Using a
    disk of that radius. This is an upper bound on what any model can reach
    when its output width differs from the label width, assuming perfect
    localisation - which is generous.
    """
    w = 2.0 * np.percentile(ndimage.distance_transform_edt(fg)[fg], 99) if fg.sum() else 0.0
    if w <= 0:
        return float("nan")
    d = (w_pred - w) / 2.0
    # NB: scipy treats iterations < 1 as "repeat until convergence", which
    # erodes the mask to nothing. Always pass >= 1, and treat |d| < 0.5 as no
    # change.
    if abs(d) < 0.5:
        pred = fg
    elif d > 0:
        pred = ndimage.binary_dilation(fg, iterations=max(1, int(round(d))))
    else:
        pred = ndimage.binary_erosion(fg, iterations=max(1, int(round(-d))))
        if pred.sum() == 0:
            return 0.0
    inter = np.logical_and(pred, fg).sum()
    union = np.logical_or(pred, fg).sum()
    return float(inter) / float(union) if union else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred-widths", default="2,3,4,6,8,12")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pred_widths = [float(x) for x in args.pred_widths.split(",")]

    rows = []
    log = []

    def emit(s):
        print(s)
        log.append(s)

    for kind, fg_rule in (("lanes", "1..254"), ("segments", ">0")):
        for split in ("train", "val"):
            names = list_names(split)
            if len(names) > args.n:
                idx = rng.choice(len(names), args.n, replace=False)
                names = [names[i] for i in sorted(idx)]

            wm_all, wp_all, share_all, er1, er2, bnd = [], [], [], [], [], []
            cls_hist, cls_total, cls_w = {}, 0, {}
            comp_all, iou_acc = [], {w: [] for w in pred_widths}

            for nm in names:
                try:
                    m = load_mask(kind, split, nm)
                except Exception:
                    continue
                fg = ((m > 0) & (m < 255)) if kind == "lanes" else (m > 0)
                if fg.sum() == 0:
                    continue

                wm, wp = stroke_width(fg)
                wm_all.append(wm)
                wp_all.append(wp)
                share_all.append(fg.mean())
                er1.append(ndimage.binary_erosion(fg, np.ones((3, 3))).sum() / fg.sum())
                er2.append(ndimage.binary_erosion(fg, np.ones((5, 5))).sum() / fg.sum())
                bnd.append(
                    (fg & ~ndimage.binary_erosion(fg, np.ones((3, 3)))).sum() / fg.sum()
                )
                if kind == "segments":
                    _, nc = ndimage.label(fg)
                    comp_all.append(nc)
                else:
                    # The lane mask is multi-class (lane marking type). The
                    # dataset collapses 1..254 to foreground, so the head is
                    # being asked to paint possibly geometrically different
                    # structures with one label. Measure both how
                    # heterogeneous the foreground is and whether the classes
                    # differ in width - if one class is a 2 px line and
                    # another an 8 px band, no single-width predictor can
                    # score well on both.
                    vals, cnts = np.unique(m[fg], return_counts=True)
                    for v, c in zip(vals, cnts):
                        vi = int(v)
                        cls_hist[vi] = cls_hist.get(vi, 0) + int(c)
                        cls_total += int(c)
                        sub = m == v
                        if sub.sum() >= 50:
                            _, wp = stroke_width(sub)
                            cls_w.setdefault(vi, []).append(wp)
                # Only meaningful for a stroke-shaped target. For DA the
                # "width" is the blob itself (~200 px) and this would mean
                # hundreds of erosion iterations per image for no information.
                if kind == "lanes":
                    for w in pred_widths:
                        iou_acc[w].append(concentric_iou(fg, w))

            if not wp_all:
                continue

            row = {
                "task": "lane" if kind == "lanes" else "da",
                "split": split,
                "n_images": len(wp_all),
                "fg_share": round(float(np.mean(share_all)), 6),
                "width_mean_native_px": round(float(np.nanmean(wm_all)), 3),
                "width_p99_native_px": round(float(np.nanmean(wp_all)), 3),
                "width_p99_median_native_px": round(float(np.nanmedian(wp_all)), 3),
                "width_p99_modelinput_px": round(float(np.nanmean(wp_all)) * INPUT_SCALE, 3),
                "erode_survive_k1": round(float(np.mean(er1)), 4),
                "erode_survive_k2": round(float(np.mean(er2)), 4),
                "boundary_share": round(float(np.mean(bnd)), 4),
            }
            if kind == "segments":
                row["n_components"] = round(float(np.mean(comp_all)), 2)
            else:
                top = sorted(cls_hist.items(), key=lambda kv: -kv[1])[:8]
                row["n_lane_classes"] = len(cls_hist)
                row["top_class_share"] = round(top[0][1] / max(cls_total, 1), 4) if top else 0.0
                row["top_class_value"] = top[0][0] if top else -1
                row["class_hist_top8"] = ";".join(
                    f"{v}:{c/max(cls_total,1):.3f}" for v, c in top
                )
            for w in pred_widths:
                row[f"iou_if_pred_w{int(w)}"] = round(
                    float(np.nanmean(iou_acc[w])), 4
                )
            rows.append(row)

            emit(
                f"{kind:9s} {split:5s} n={len(wp_all):4d}  "
                f"width_p99={np.nanmean(wp_all):6.2f}px native "
                f"({np.nanmean(wp_all)*INPUT_SCALE:5.2f}px at input)  "
                f"fg_share={np.mean(share_all):.5f}  "
                f"erode_k1={np.mean(er1):.4f} erode_k2={np.mean(er2):.4f}  "
                f"boundary={np.mean(bnd):.3f}"
            )
            if kind == "lanes":
                s = "   perfect-localisation IoU vs predicted stroke width: "
                s += "  ".join(
                    f"w{int(w)}={np.nanmean(iou_acc[w]):.3f}" for w in pred_widths
                )
                emit(s)
                if cls_hist:
                    top = sorted(cls_hist.items(), key=lambda kv: -kv[1])[:8]
                    emit(
                        "   foreground is "
                        f"{len(cls_hist)} distinct lane-marking classes; "
                        "share by class: "
                        + "  ".join(
                            f"{v}={c/max(cls_total,1):.3f}" for v, c in top
                        )
                    )
                    if cls_w:
                        emit("   per-class stroke width (p99, native px):")
                        for v, _ in top:
                            if v in cls_w and len(cls_w[v]) >= 5:
                                emit(
                                    f"     class {v:3d}  share="
                                    f"{cls_hist[v]/max(cls_total,1):.3f}  "
                                    f"width={np.nanmean(cls_w[v]):5.2f}px  "
                                    f"n={len(cls_w[v])}"
                                )

    emit("")
    emit("Reading")
    emit("-------")
    tr = next((r for r in rows if r["task"] == "lane" and r["split"] == "train"), None)
    va = next((r for r in rows if r["task"] == "lane" and r["split"] == "val"), None)
    if tr and va:
        ratio = va["width_p99_native_px"] / max(tr["width_p99_native_px"], 1e-9)
        bound = min(ratio, 1.0 / ratio) if ratio else float("nan")
        emit(
            f"lane stroke width: train {tr['width_p99_native_px']:.2f} px, "
            f"val {va['width_p99_native_px']:.2f} px, ratio {ratio:.3f}"
        )
        emit(
            f"A model trained on the train stroke and evaluated on the val "
            f"stroke, localising perfectly, is bounded at IoU {bound:.3f} by "
            f"the width mismatch alone."
        )
        if bound < 0.9:
            emit(
                "=> The train/val stroke widths differ. Part of the lane "
                "deficit is the label, not the model. H8 is live."
            )
        else:
            emit(
                "=> Train and val strokes match. The lane deficit is not a "
                "train/val width mismatch. H8 is not supported on this axis."
            )
        emit(
            f"At model input the val stroke is "
            f"{va['width_p99_modelinput_px']:.2f} px wide and the smallest "
            f"addressable unit at 1/8 is 8 px of input. That is the geometry "
            f"behind the 1/8 block-fill reference of 0.1853."
        )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log) + "\n")
    emit("")
    emit(f"wrote {OUT_CSV}")
    emit(f"wrote {OUT_LOG}")


if __name__ == "__main__":
    main()
