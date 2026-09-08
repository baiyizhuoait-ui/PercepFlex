"""Phase 5 STEP 2 - where does DA actually fail, and is it the labels' fault?

Correction that motivates the rewrite
-------------------------------------
Level 2 reported a far-field DA collapse (top third IoU 0.393 for r2_z16
against 0.763 mid-field) and read it as the dominant DA error. That number is
contaminated. BDD100K frames are 1280x720 and are letterboxed to 640x640 with
pad = (w=0, h=140), so the image content occupies rows 140..500 and the top
third of the 640 canvas is 66% grey padding. Measured directly, the ground
truth has 0.0% of its drivable area in rows 0..213. The "far field" band was
mostly measuring how each model behaves on letterbox padding, and only a small
self-selected subset of images (those where distant road reaches into rows
140..213) contributed at all. That also explains why the top band swung
wildly and non-monotonically with z width while mid and bottom were stable.

This version crops to the content region first, exactly as the official
evaluation does, and then bands the content into thirds. It also reports an
"all" row so the pipeline can be checked against the published da_fg numbers.

Question
--------
H11     The far field is semantically hard: distant road is a few pixels tall,
        occluded, and needs context rather than local appearance.
H11-alt It is an evaluation artefact - distant road owns few pixels, so the
        metric is simply unforgiving there.

Normalising each band against a reference strategy separates them:

  blockfill      fg IoU(gt_band, upsample(maxpool(gt, 8))_band). NAMED
                 HONESTLY: this is the score of the strategy "paint every 1/8
                 cell the target touches", not a strict upper bound. For
                 targets thinner than one cell it is effectively a ceiling;
                 for blobs larger than a cell a model that upsamples
                 bilinearly to full resolution can beat it. Level 2 called it
                 a ceiling. That is only correct for lane.
  delta          band IoU - blockfill.
  mixed blocks   share of 1/8 blocks in the band containing BOTH drivable and
                 non-drivable ground truth. Unresolvable at 1/8.
  boundary share share of band foreground within 1 px of the boundary.
  gt share       share of the image's DA ground truth in this band - also the
                 band's weight in the loss (H12).

Cells include R0 and the probe-C rebalanced models, because if rebalancing the
gradient recovers the far field, the far-field deficit is a sharing effect; if
it does not, it is something else. Those checkpoints already exist, so adding
them costs no training.
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

from datasets.bdd100k import BDD100KDataset, collate_train            # noqa: E402
from models.static_model import StaticMultiTaskModel                  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "phase5", "phase5_da_analysis.csv")

CELLS = [
    ("r0", "r0_z16",  "configs/phase3a_ebase_z16.yaml",
     "experiments/phase3a/exp3A_ebase_z16/checkpoint.pt"),
    ("r2", "r2_z16",  "configs/phase4a_r2_z16.yaml",
     "experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt"),
    ("r2", "r2_z32",  "configs/phase4a_r2_z32.yaml",
     "experiments/phase4a/exp4A_r2_z32_e20/checkpoint.pt"),
    ("r2", "r2_z128", "configs/phase4a_r2_z128.yaml",
     "experiments/phase4a/exp4A_r2_z128_e20/checkpoint.pt"),
    ("r2rw", "rw_z16", "configs/phase4c_rw_z16.yaml",
     "experiments/phase4a/exp4C_rw_z16_e20/checkpoint.pt"),
    ("r2rw", "rw_z32", "configs/phase4c_rw_z32.yaml",
     "experiments/phase4a/exp4C_rw_z32_e20/checkpoint.pt"),
]
BANDS = ["far", "mid", "near", "all"]
RES_FACTOR = 8  # Z resolution; the head upsamples to 640 but OPERATES at 1/8


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def fg_iou(pred, gt):
    inter = (pred & gt).float().sum()
    union = pred.float().sum() + gt.float().sum() - inter
    return (inter / union).item() if union > 0 else float("nan")


def erode(m, k):
    return -torch.nn.functional.max_pool2d(-m, 2 * k + 1, 1, k)


def equal_mass_bands(g):
    """Split the content rows so each band holds ~1/3 of the DA ground truth.

    Fixed thirds of the image do not work here. After cropping the letterbox
    the content is 360 rows tall and its top third is essentially sky: only 7
    of 285 images had any drivable-area ground truth there, which makes a
    fixed 'far' band unmeasurable. Banding by equal ground-truth mass instead
    guarantees every band is populated and makes the three bands directly
    comparable, because each carries the same share of the loss.
    """
    per_row = g[0, 0].sum(dim=1)
    total = per_row.sum().item()
    h = per_row.numel()
    if total <= 0:
        return None
    cum = torch.cumsum(per_row, 0)
    i1 = int((cum >= total / 3.0).nonzero()[0])
    i2 = int((cum >= 2.0 * total / 3.0).nonzero()[0])
    i2 = max(i2, i1 + 1)
    return {"far": slice(0, i1 + 1), "mid": slice(i1 + 1, i2 + 1),
            "near": slice(i2 + 1, h), "all": slice(0, h)}, (i1, i2, h)


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
            print("[skip] %s (no checkpoint)" % cell)
            continue
        cfg = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
        model = build(cfg).to(dev).eval()
        sd = torch.load(ck, map_location="cpu")["model_state"]
        model.load_state_dict(sd, strict=False)

        ds = BDD100KDataset(cfg["data"]["root"], split="tri_val")
        dl = torch.utils.data.DataLoader(
            ds, batch_size=args.bs, shuffle=False, num_workers=2,
            collate_fn=collate_train, drop_last=True)

        keys = ["iou", "ceil", "prec", "rec", "area", "n"]
        acc = {b: {k: [] for k in keys} for b in BANDS}
        struct = {b: {"mixed": [], "bnd": [], "gshare": []}
                  for b in ["far", "mid", "near"]}
        band_row_pos = []
        seen = 0
        with torch.no_grad():
            for batch in dl:
                if seen >= args.n:
                    break
                bs = batch["image"].shape[0]
                seen += bs
                x = batch["image"].to(dev)
                out = model(x)
                da_lg = out[1] if isinstance(out, tuple) else out["da"]
                pred = da_lg.argmax(1, keepdim=True).bool().cpu()

                for i in range(bs):
                    ph = int(batch["pad"][i][1])
                    nh = int(batch["content_size"][i][0])
                    top, bot = ph, ph + nh
                    g = (batch["da_mask"][i:i + 1, :, top:bot, :] > 0.5)
                    p = pred[i:i + 1, :, top:bot, :]
                    if g.sum().item() < 10:
                        continue

                    f = RES_FACTOR
                    up = torch.nn.functional.interpolate(
                        torch.nn.functional.max_pool2d(g.float(), f, f),
                        size=g.shape[-2:], mode="nearest") > 0.5
                    avg = torch.nn.functional.avg_pool2d(g.float(), f, f)
                    mixed = ((avg > 0.0) & (avg < 1.0)).float()
                    e1 = erode(g.float(), 1) > 0.5
                    bnd_share = 1.0 - e1.sum().item() / max(1, g.sum().item())

                    bands = equal_mass_bands(g)
                    if bands is None:
                        continue
                    sl, (i1, i2, hh) = bands
                    band_row_pos.append((i1 / hh, i2 / hh))
                    for bname in BANDS:
                        s = sl[bname]
                        gb, pb, ub = g[:, :, s, :], p[:, :, s, :], up[:, :, s, :]
                        inter = (pb & gb).float().sum().item()
                        psum = pb.float().sum().item()
                        gsum = gb.float().sum().item()
                        den = psum + gsum - inter
                        if gsum < 1 or den <= 0:
                            continue
                        acc[bname]["iou"].append(inter / den)
                        acc[bname]["ceil"].append(fg_iou(ub, gb))
                        acc[bname]["prec"].append(inter / max(1, psum))
                        acc[bname]["rec"].append(inter / max(1, gsum))
                        acc[bname]["area"].append(psum / gsum)
                        acc[bname]["n"].append(1)
                        if bname in struct:
                            blk = mixed[:, :, max(0, s.start // f):
                                        max(1, s.stop // f), :]
                            struct[bname]["mixed"].append(
                                blk.sum().item() / max(1, blk.numel()))
                            e1b = erode(gb.float(), 1) > 0.5
                            struct[bname]["bnd"].append(
                                1.0 - e1b.sum().item() / max(1, gsum))
                            struct[bname]["gshare"].append(
                                gsum / max(1, g.sum().item()))

        def m(v):
            v = [x for x in v if x == x]
            return float(np.mean(v)) if v else float("nan")

        for bname in BANDS:
            a = acc[bname]
            if not a["iou"]:
                continue
            row = {
                "cell": cell, "variant": variant, "band": bname,
                "da_fg_iou": round(m(a["iou"]), 4),
                "blockfill_1over8_iou": round(m(a["ceil"]), 4),
                "iou_minus_blockfill": round(m(a["iou"]) - m(a["ceil"]), 4),
                "precision": round(m(a["prec"]), 4),
                "recall": round(m(a["rec"]), 4),
                "pred_over_gt_area": round(m(a["area"]), 4),
                "n_valid_images": len(a["iou"]),
                "n_images": seen,
            }
            if bname in struct:
                s = struct[bname]
                row["mixed_block_share"] = round(m(s["mixed"]), 4)
                row["boundary_pixel_share"] = round(m(s["bnd"]), 4)
                row["gt_pixel_share"] = round(m(s["gshare"]), 4)
            else:
                row["mixed_block_share"] = ""
                row["boundary_pixel_share"] = ""
                row["gt_pixel_share"] = ""
            rows.append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["cell", "variant", "band", "da_fg_iou", "blockfill_1over8_iou",
            "iou_minus_blockfill", "precision", "recall", "pred_over_gt_area",
            "mixed_block_share", "boundary_pixel_share", "gt_pixel_share",
            "n_valid_images", "n_images"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print("\nDA by band of the CONTENT region (letterbox padding removed):")
    print("%-9s %-6s %8s %10s %9s %8s %8s %8s %7s" %
          ("cell", "band", "fg_iou", "blockfill", "delta", "prec", "rec",
           "area", "nvalid"))
    for r in rows:
        print("%-9s %-6s %8.4f %10.4f %9.4f %8.4f %8.4f %8.2f %7d" %
              (r["cell"], r["band"], r["da_fg_iou"],
               r["blockfill_1over8_iou"], r["iou_minus_blockfill"],
               r["precision"], r["recall"], r["pred_over_gt_area"],
               r["n_valid_images"]))
    print("")
    if band_row_pos:
        a = np.array(band_row_pos)
        print("Band boundaries (fraction of content height, equal GT mass):")
        print("  far|mid  at %.3f   mid|near at %.3f" % (a[:, 0].mean(), a[:, 1].mean()))
    print("Structural difficulty per band (model independent):")
    for r in rows:
        if r["band"] in ("far", "mid", "near") and r["cell"] == rows[0]["cell"]:
            print("  %-5s gt_share=%.4f  mixed_block=%.4f  boundary_px=%.4f"
                  % (r["band"], float(r["gt_pixel_share"]),
                     float(r["mixed_block_share"]),
                     float(r["boundary_pixel_share"])))
    print("")
    print("WROTE %s" % OUT)


if __name__ == "__main__":
    main()
