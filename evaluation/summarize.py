#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize baseline evaluation results from experiments/exp_001/ into a table."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments", "exp_001")
    rows = []
    for d in sorted(os.listdir(base)):
        jp = os.path.join(base, d, "metrics.json")
        if os.path.exists(jp):
            with open(jp) as f:
                m = json.load(f)
            rows.append(m)
    if not rows:
        print("no metrics found")
        return
    hdr = ["model", "params(M)", "FLOPs(G)", "size(MB)", "mAP50", "mAP50-95",
           "DA_acc", "DA_fgIoU", "DA_mIoU", "Lane_acc", "Lane_fgIoU", "Lane_mIoU",
           "Lane_lineAcc", "lat(ms)", "FPS", "mem(MiB)"]
    print(" | ".join(hdr))
    print("-|-".join([""] * len(hdr)))
    for m in rows:
        print(" | ".join([
            str(m.get("model", "")),
            f"{m.get('parameters', 0) / 1e6:.3f}",
            f"{m.get('flops', 0) / 1e9:.3f}",
            str(m.get("model_size_mb", "")),
            f"{m.get('mAP50', 0):.4f}",
            f"{m.get('mAP50_95', 0):.4f}",
            f"{m.get('da_pixel_acc', 0):.4f}",
            f"{m.get('da_fg_iou', 0):.4f}",
            f"{m.get('da_mIoU', 0):.4f}",
            f"{m.get('lane_pixel_acc', 0):.4f}",
            f"{m.get('lane_fg_iou', 0):.4f}",
            f"{m.get('lane_mIoU', 0):.4f}",
            f"{m.get('lane_line_acc', 0):.4f}",
            str(m.get("avg_latency_ms", "")),
            str(m.get("fps", "")),
            str(m.get("gpu_memory_mib", "")),
        ]))


if __name__ == "__main__":
    main()
