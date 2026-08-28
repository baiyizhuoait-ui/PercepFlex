#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDD100K dataset verification + split manifest builder.

Verifies the three-task annotation↔image correspondence and writes the
per-task usable image lists used by Phase 1 training/evaluation:

  data/bdd100k/splits/
    det_train.txt  det_val.txt    (from labels/*.json)
    da_train.txt   da_val.txt     (from segments/masks/*, intersected with images)
    lane_train.txt lane_val.txt   (from lanes/masks/*, intersected with images)
    tri_train.txt  tri_val.txt    (three-task intersection; YOLOP-style training protocol)

Usage:
    gpu_env/bin/python scripts/verify_dataset.py [--data-dir trac/data/bdd100k]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def basename_set(files, ext=None):
    return {os.path.splitext(x)[0] for x in files if ext is None or x.endswith(ext)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "bdd100k"))
    args = ap.parse_args()
    d = args.data_dir

    img_tr = basename_set(os.listdir(os.path.join(d, "images/100k/train")), ".jpg")
    img_va = basename_set(os.listdir(os.path.join(d, "images/100k/val")), ".jpg")
    lane_tr = basename_set(os.listdir(os.path.join(d, "lanes/masks/train")), ".png")
    lane_va = basename_set(os.listdir(os.path.join(d, "lanes/masks/val")), ".png")
    seg_tr = basename_set(os.listdir(os.path.join(d, "segments/masks/train")), ".png")
    seg_va = basename_set(os.listdir(os.path.join(d, "segments/masks/val")), ".png")

    print(f"images   train={len(img_tr)} val={len(img_va)}")
    print(f"lane     train={len(lane_tr)} (match {len(lane_tr & img_tr)}) "
          f"val={len(lane_va)} (match {len(lane_va & img_va)})")
    print(f"seg(DA)  train={len(seg_tr)} (match img_train {len(seg_tr & img_tr)}, "
          f"img_val {len(seg_tr & img_va)})")
    print(f"         val={len(seg_va)} (match img_val {len(seg_va & img_va)})")

    # detection labels
    det_tr, det_va = set(), set()
    for split, out in [("train", det_tr), ("val", det_va)]:
        jpath = os.path.join(d, "labels", f"bdd100k_labels_images_{split}.json")
        with open(jpath) as f:
            data = json.load(f)
        out.update(x["name"].rsplit(".", 1)[0] for x in data)
        print(f"det json {split}: {len(data)} entries")

    # usable per-task lists
    da_tr = seg_tr & img_tr
    da_va = (seg_tr & img_va) | (seg_va & img_va)  # val images that have any DA mask
    lane_tr_u = lane_tr & img_tr
    lane_va_u = lane_va & img_va
    det_tr_u = det_tr & img_tr
    det_va_u = det_va & img_va

    tri_tr = da_tr & lane_tr_u & det_tr_u
    tri_va = da_va & lane_va_u & det_va_u

    print(f"\nusable per-task  train / val")
    print(f"  det : {len(det_tr_u)} / {len(det_va_u)}")
    print(f"  da  : {len(da_tr)} / {len(da_va)}")
    print(f"  lane: {len(lane_tr_u)} / {len(lane_va_u)}")
    print(f"  three-task intersection: {len(tri_tr)} / {len(tri_va)}")

    split_dir = os.path.join(d, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for name, s in [("det_train", det_tr_u), ("det_val", det_va_u),
                    ("da_train", da_tr), ("da_val", da_va),
                    ("lane_train", lane_tr_u), ("lane_val", lane_va_u),
                    ("tri_train", tri_tr), ("tri_val", tri_va)]:
        with open(os.path.join(split_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(sorted(s)) + ("\n" if s else ""))
    print(f"\nmanifests written to {split_dir}/")


if __name__ == "__main__":
    main()
