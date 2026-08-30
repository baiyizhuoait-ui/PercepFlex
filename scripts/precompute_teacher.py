#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute teacher targets for OFFLINE knowledge distillation.

Runs a frozen teacher over a training subset ONCE (no grad), caching:
  - feature_1_8 (downsampled to 1/16 for storage, fp16)
  - da / lane logits at 1/8 (fp16)
so the KD student training reads from disk instead of running the teacher
every step (Plan A — ~10x faster, no GPU memory pressure).

Usage:
    gpu_env/bin/python scripts/precompute_teacher.py --teacher YOLOP --num-images 10000
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datasets.bdd100k import BDD100KDataset  # noqa: E402
from distillation.kd import TeacherAdapter  # noqa: E402


def build_teacher(name, device):
    if name == "YOLOP":
        from evaluation.evaluate_baseline import build_yolop
        return build_yolop(device), "imagenet"
    from load_baseline_weights import load_baseline
    preset = "large" if name == "TwinLiteNetPlus" else "base"
    t = load_baseline(name, preset, device)
    norm = "imagenet" if name == "TriLiteNet" else "unit"
    return t, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    choices=["YOLOP", "TwinLiteNetPlus", "TriLiteNet"])
    ap.add_argument("--num-images", type=int, default=10000)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "bdd100k"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "expF_single_teacher", "kd_cache"))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher, norm = build_teacher(args.teacher, device)
    teacher.eval()
    adapter = TeacherAdapter(args.teacher, teacher, norm).to(device)

    ds = BDD100KDataset(args.data_root, split="tri_train", with_det=False)
    ds.names = ds.names[: args.num_images]
    names = ds.names
    print(f"[precompute] teacher={args.teacher} images={len(names)}", flush=True)

    # probe shapes with one image
    x = ds[0]["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        feat, da, lane = adapter(x)
    fc, fh, fw = feat.shape[1], feat.shape[2], feat.shape[3]
    print(f"[precompute] feat {fc}x{fh}x{fw}, da {tuple(da.shape)}", flush=True)
    # storage: feature at 1/16 (or native if already small), outputs at 1/8
    fh16, fw16 = fh // 2, fw // 2
    dh8, dw8 = da.shape[2] // 8, da.shape[3] // 8

    feat_mem = np.lib.format.open_memmap(
        os.path.join(args.out, f"{args.teacher}_feat.npy"), mode="w+",
        dtype=np.float16, shape=(len(names), fc, fh16, fw16))
    da_mem = np.lib.format.open_memmap(
        os.path.join(args.out, f"{args.teacher}_da.npy"), mode="w+",
        dtype=np.float16, shape=(len(names), 2, dh8, dw8))
    lane_mem = np.lib.format.open_memmap(
        os.path.join(args.out, f"{args.teacher}_lane.npy"), mode="w+",
        dtype=np.float16, shape=(len(names), 2, dh8, dw8))

    for i in range(0, len(names), args.batch):
        idx = list(range(i, min(i + args.batch, len(names))))
        imgs = torch.stack([ds[j]["image"] for j in idx]).to(device)
        with torch.no_grad():
            feat, da, lane = adapter(imgs)
        feat = torch.nn.functional.interpolate(feat, size=(fh16, fw16),
                                               mode="bilinear", align_corners=False)
        da = torch.nn.functional.interpolate(da, size=(dh8, dw8),
                                             mode="bilinear", align_corners=False)
        lane = torch.nn.functional.interpolate(lane, size=(dh8, dw8),
                                               mode="bilinear", align_corners=False)
        feat_mem[idx] = feat.cpu().half().numpy()
        da_mem[idx] = da.cpu().half().numpy()
        lane_mem[idx] = lane.cpu().half().numpy()
        if (i // args.batch) % 10 == 0:
            print(f"  [{i}/{len(names)}]", flush=True)

    with open(os.path.join(args.out, f"{args.teacher}_manifest.txt"), "w") as f:
        f.write("\n".join(names))
    print(f"[precompute] saved -> {args.out}/{args.teacher}_*.npy (+ manifest)", flush=True)


if __name__ == "__main__":
    main()
