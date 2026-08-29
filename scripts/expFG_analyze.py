#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment F/G analysis: Single-Teacher Cross-Architecture KD comparison.

No KD (exp_train_A) vs +YOLOP / +TwinLiteNetPlus / +TriLiteNet teacher KD.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("no_kd", "experiments/exp_train_A/eval_1000", "0.235M"),
    ("kd_yolop", "experiments/expF_single_teacher/kd_yolop_eval", "0.235M"),
    ("kd_twinlite", "experiments/expF_single_teacher/kd_twinlite_eval", "0.235M"),
    ("kd_trilite", "experiments/expF_single_teacher/kd_trilite_eval", "0.235M"),
]


def main():
    print(f"{'model':<14} {'Params':>8} {'mAP50':>7} {'DA_mIoU':>8} {'Lane_fgIoU':>10}")
    rows = []
    for tag, path, params in MODELS:
        p = os.path.join(ROOT, path, "metrics.json")
        if not os.path.exists(p):
            print(f"{tag:<14} {params:>8}  (training/eval pending)")
            continue
        m = json.load(open(p))
        rows.append({
            "model": tag, "params": params,
            "mAP50": m.get("mAP50", 0), "da_mIoU": m.get("da_mIoU", 0),
            "lane_fg_iou": m.get("lane_fg_iou", 0),
        })
        print(f"{tag:<14} {params:>8} {m.get('mAP50',0):>7.4f} {m.get('da_mIoU',0):>8.4f} "
              f"{m.get('lane_fg_iou',0):>10.4f}")
    if len(rows) >= 2:
        base = rows[0]
        print("\nKD effect (vs no_kd):")
        for r in rows[1:]:
            print(f"  {r['model']}: mAP {r['mAP50']-base['mAP50']:+.4f} "
                  f"DA {r['da_mIoU']-base['da_mIoU']:+.4f} "
                  f"Lane {r['lane_fg_iou']-base['lane_fg_iou']:+.4f}")


if __name__ == "__main__":
    main()
