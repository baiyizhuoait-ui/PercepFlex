#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified baseline evaluation (Phase 1, Step 3).

Runs each baseline (official weights) on the BDD100K val split and reports the
full metric set required by the taskbook: Parameters, FLOPs, Model Size,
Detection mAP, DA mIoU, Lane IoU, latency (mean/P50/P95), FPS, GPU memory.

Protocol (matches official light-model val pipelines):
- input: 640x640 letterbox (pad cropped for segmentation metrics)
- detection: single-class ('car') mAP@0.5 and mAP@0.5:0.95, NMS conf=0.001/iou=0.6
- DA / Lane: binary metrics (pixel Acc, foreground IoU, 2-class mIoU;
  lane also lineAccuracy=(sens+spec)/2)
- normalization: ImageNet mean/std for YOLOP & TriLiteNet, unit for
  TwinLiteNet / TwinLiteNetPlus (their official preprocessing)

Usage:
    gpu_env/bin/python evaluation/evaluate_baseline.py --baseline TriLiteNet --preset tiny
    gpu_env/bin/python evaluation/evaluate_baseline.py --baseline YOLOP
    gpu_env/bin/python evaluation/evaluate_baseline.py --baseline TwinLiteNetPlus --preset large
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

from datasets.bdd100k import BDD100KDataset  # noqa: E402
from evaluation.metrics import evaluate_detection, SegmentationMetric  # noqa: E402
from evaluation.nms import non_max_suppression  # noqa: E402
from profiling.benchmark import profile_model  # noqa: E402
from load_baseline_weights import load_twinlitenet, load_twinlitenetplus, load_trilitenet  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_yolop(device):
    sys.path.insert(0, os.path.join(ROOT, "..", "YOLOP"))
    from lib.config import cfg  # noqa: E402
    from lib.models import get_net  # noqa: E402

    cfg.defrost()
    cfg.MODEL.NAME = "YOLOP"
    cfg.MODEL.IMAGE_SIZE = [640, 640]
    cfg.freeze()
    model = get_net(cfg)
    sd = torch.load(os.path.join(ROOT, "..", "YOLOP", "weights", "End-to-end.pth"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd)
    return model.to(device)


def build_ours(model_cls_name, weights, device):
    """Build our model (StaticMultiTaskModel / AdaptiveMultiTaskModel) from config."""
    import yaml
    from models.static_model import StaticMultiTaskModel
    from models.adaptive_model import AdaptiveMultiTaskModel
    cfg_path = os.path.join(ROOT, "configs", "ours_static.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["model"]
    if model_cls_name == "OursDynamic" and weights:
        # detect shared vs task-wise router from the checkpoint
        probe = torch.load(weights, map_location="cpu", weights_only=False)
        probe = probe.get("model_state", probe)
        rw = probe.get("router.mlp.2.weight")
        if rw is not None and rw.shape[0] == 3:
            cfg.setdefault("router", {})["shared"] = True
            print("[ours] detected SHARED router")
        if "router.diff_head.weight" in probe:
            cfg.setdefault("router", {})["type"] = "difficulty"
            print("[ours] detected DIFFICULTY router")
    cls = StaticMultiTaskModel if model_cls_name == "OursStatic" else AdaptiveMultiTaskModel
    model = cls(cfg)
    if weights:
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("model_state", sd)
        cur = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(compat, strict=False)
        print(f"[ours] loaded {weights}: {len(compat)} keys missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device), "unit"


def build(name, preset, device):
    if name == "YOLOP":
        return build_yolop(device), "imagenet"
    if name == "TwinLiteNet":
        return load_twinlitenet(device), "unit"
    if name == "TwinLiteNetPlus":
        return load_twinlitenetplus(preset or "nano", device), "unit"
    if name == "TriLiteNet":
        return load_trilitenet(preset or "tiny", device), "imagenet"
    if name in ("OursStatic", "OursDynamic"):
        return build_ours(name, preset, device)  # preset = weights path
    raise ValueError(name)


def forward_outs(model, name, x):
    """Return (det_logits_or_None, da_logits_or_None, lane_logits_or_None)."""
    out = model(x)
    if name in ("YOLOP", "TriLiteNet"):
        det_tuple, da, lane = out
        det = det_tuple[0]  # (1, 25200, 6) decoded
        return det, da, lane
    if name in ("OursStatic", "OursDynamic"):
        det, da, lane = out
        return det, da, lane
    # TwinLiteNet / TwinLiteNetPlus: (da, lane)
    da, lane = out
    return None, da, lane


def preprocess(img, norm):
    x = img.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    if norm == "imagenet":
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    return x


def crop_content(mask, item):
    """Crop letterbox pad from a (1,H,W) mask using item meta."""
    pad_w, pad_h = item["pad"]
    nh, nw = item["content_size"]
    return mask[:, pad_h:pad_h + nh, pad_w:pad_w + nw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    choices=["YOLOP", "TwinLiteNet", "TwinLiteNetPlus", "TriLiteNet", "OursStatic", "OursDynamic"])
    ap.add_argument("--preset", default=None, help="nano/small/medium/large (TLP) or tiny/small/base (TriLiteNet)")
    ap.add_argument("--split", default="tri_val")
    ap.add_argument("--num-images", type=int, default=0, help="0 = full split")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "bdd100k"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--alloc-stats", action="store_true")
    ap.add_argument("--force-widths", default=None,
                    help="comma widths det,da,lane e.g. 0.25,0.25,0.25 (OursDynamic)")
    ap.add_argument("--router-mode", default="learned",
                    choices=["learned", "random", "fixed"],
                    help="routing policy for OursDynamic (Exp3)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, norm = build(args.baseline, args.preset, device)
    model.eval()
    name = f"{args.baseline}" + (f"-{args.preset}" if args.preset else "")

    # ---------------- profiling ----------------
    profile = None
    if not args.no_profile:
        try:
            if args.baseline in ("OursStatic", "OursDynamic"):
                profile = _profile_ours(model, (1, 3, 640, 640), device, 3)
            else:
                profile = profile_model(model, device="cuda" if device.type == "cuda" else "cpu")
            print(f"[profile] params={profile['parameters']/1e6:.3f}M "
                  f"flops={profile['flops']/1e9:.3f}G "
                  f"lat={profile['mean_ms']:.2f}ms fps={profile['fps']:.1f} "
                  f"mem={profile['peak_gpu_memory_mib']}MiB")
        except Exception as e:  # noqa: BLE001
            print(f"[profile] failed: {e}")

    # ---------------- dataset ----------------
    ds = BDD100KDataset(args.data_root, split=args.split, with_det=(name != "TwinLiteNet"
                                                                    and not name.startswith("TwinLiteNetPlus")),
                        with_da=True, with_lane=True)
    if args.num_images:
        ds.names = ds.names[:args.num_images]
        if getattr(ds, "_det_by_name", None):
            keep = set(ds.names)
            ds._det_by_name = {k: v for k, v in ds._det_by_name.items() if k in keep}
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # ---------------- eval loop ----------------
    det_preds, det_gts = [], []
    da_metric = SegmentationMetric(2)   # incremental confusion matrices (O(1) memory)
    lane_metric = SegmentationMetric(2)
    alloc_rows = []                     # per-frame router assignments (dynamic models)
    t_forward, t_total = 0.0, time.time()
    with torch.no_grad():
        for it, item in enumerate(loader):
            img = item["image"][0]
            x = preprocess(img, norm)
            t0 = time.time()
            if args.baseline == "OursDynamic":
                fw = None
                if args.force_widths:
                    fw = torch.tensor([float(v) for v in args.force_widths.split(",")])
                elif args.router_mode == "random":
                    # Exp3: random per-frame task-wise allocation (uniform over budgets)
                    budgets = torch.tensor([0.25, 0.5, 1.0])
                    idx = torch.randint(0, 3, (1, 3))
                    fw = budgets[idx].squeeze(0)
                elif args.router_mode == "fixed":
                    fw = torch.tensor([0.5, 0.5, 0.5])  # mid budget, no routing
                det_logits, da_logits, lane_logits, routing = \
                    model(x, hard=True, return_routing=True, force_widths=fw)
                if args.alloc_stats:
                    alloc_rows.append({
                        "frame": item["name"][0],
                        "det": int(routing["assignments"][0, 0].item()),
                        "da": int(routing["assignments"][0, 1].item()),
                        "lane": int(routing["assignments"][0, 2].item()),
                        "w_det": float(routing["widths"][0, 0].item()),
                        "w_da": float(routing["widths"][0, 1].item()),
                        "w_lane": float(routing["widths"][0, 2].item()),
                    })
            else:
                det_logits, da_logits, lane_logits = forward_outs(model, args.baseline, x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forward += time.time() - t0

            if det_logits is not None:
                dets = non_max_suppression(det_logits, conf_thres=args.conf, iou_thres=args.iou)[0]
                if len(dets):
                    det_preds.append((dets[:, :4].cpu(), dets[:, 4].cpu(), dets[:, 5].long().cpu()))
                else:
                    det_preds.append(None)
                det_gts.append(item["det_targets"][0][:, 1:] * 640)  # xywh norm -> xyxy px
                det_gts[-1] = torch.cat([det_gts[-1][:, :2] - det_gts[-1][:, 2:] / 2,
                                         det_gts[-1][:, :2] + det_gts[-1][:, 2:] / 2], 1) if len(det_gts[-1]) else det_gts[-1]

            # segmentation (crop pad, argmax over 2 channels), accumulate confusion matrix
            pad_w, pad_h = item["pad"][0].item(), item["pad"][1].item()
            nh, nw = item["content_size"][0].item(), item["content_size"][1].item()
            if da_logits is not None:
                da_pred = da_logits[0].argmax(0)[pad_h:pad_h + nh, pad_w:pad_w + nw].cpu().numpy()
                da_gt = item["da_mask"][0][0, pad_h:pad_h + nh, pad_w:pad_w + nw].numpy().astype(np.uint8)
                da_metric.add_batch(da_pred, da_gt)
            if lane_logits is not None:
                lane_pred = lane_logits[0].argmax(0)[pad_h:pad_h + nh, pad_w:pad_w + nw].cpu().numpy()
                lane_gt = item["lane_mask"][0][0, pad_h:pad_h + nh, pad_w:pad_w + nw].numpy().astype(np.uint8)
                lane_metric.add_batch(lane_pred, lane_gt)
            if (it + 1) % 1000 == 0:
                print(f"  [{it+1}/{len(ds)}] fwd {t_forward/(it+1)*1e3:.1f}ms/img "
                      f"elapsed {time.time()-t_total:.0f}s", flush=True)

    print(f"[eval] {len(ds)} images, forward {t_forward/len(ds)*1e3:.1f}ms/img (CUDA-synced), "
          f"total {time.time()-t_total:.0f}s")

    # ---------------- metrics ----------------
    metrics = {"model": name, "split": args.split, "input_size": [640, 640],
               "eval_fwd_ms": round(t_forward / max(len(ds), 1) * 1e3, 3),
               "eval_fps": round(1000.0 / (t_forward / max(len(ds), 1) * 1e3), 2)}
    if det_preds:
        m = evaluate_detection(det_preds, det_gts)
        metrics.update({"mAP50": round(m["mAP50"], 4), "mAP50_95": round(m["mAP50_95"], 4),
                        "det_n_gt": m["n_gt"], "det_n_pred": m["n_pred"]})
    if da_metric.conf.sum() > 0:
        metrics.update({"da_pixel_acc": round(da_metric.pixel_accuracy(), 4),
                        "da_fg_iou": round(da_metric.fg_iou(), 4),
                        "da_mIoU": round(da_metric.m_iou(), 4)})
    if lane_metric.conf.sum() > 0:
        metrics.update({"lane_pixel_acc": round(lane_metric.pixel_accuracy(), 4),
                        "lane_fg_iou": round(lane_metric.fg_iou(), 4),
                        "lane_mIoU": round(lane_metric.m_iou(), 4),
                        "lane_line_acc": round(lane_metric.line_accuracy(), 4)})
    if alloc_rows:
        try:
            from profiling.flops_real import count_flops
            from models.static_model import StaticMultiTaskModel
            import yaml as _yaml
            with open(os.path.join(ROOT, "configs", "ours_static.yaml")) as f:
                scfg = _yaml.safe_load(f)["model"]
            sm = StaticMultiTaskModel(scfg).to(device).eval()
            x = torch.randn(1, 3, 640, 640).to(device)
            enc = count_flops(sm.encoder, x)
            heads = {}
            for w in (0.25, 0.5, 1.0):
                sm.set_width(w)
                heads[w] = (count_flops(sm, x) - enc) / 3.0
            wmap = {"0": 0.25, "1": 0.5, "2": 1.0}
            import numpy as _np
            avg = float(enc)
            for t in ("det", "da", "lane"):
                wsum = sum(wmap[r[t]] for r in alloc_rows) / len(alloc_rows)
                avg += float(_np.interp(wsum, [0.25, 0.5, 1.0],
                                        [heads[0.25], heads[0.5], heads[1.0]]))
            metrics["flops"] = avg
            metrics["avg_alloc_width"] = round(sum(
                (wmap[r["det"]] + wmap[r["da"]] + wmap[r["lane"]]) / 3 for r in alloc_rows) / len(alloc_rows), 4)
        except Exception as e:
            print(f"[alloc-flops] failed: {e}")
    if profile:
        metrics.update({"parameters": profile["parameters"], "flops": profile["flops"],
                        "model_size_mb": profile["model_size_mb"],
                        "avg_latency_ms": round(profile["mean_ms"], 3),
                        "p50_latency_ms": round(profile["p50_ms"], 3),
                        "p95_latency_ms": round(profile["p95_ms"], 3),
                        "fps": round(profile["fps"], 2),
                        "gpu_memory_mib": profile["peak_gpu_memory_mib"]})

    print(json.dumps(metrics, indent=2))

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if alloc_rows:
            import csv
            with open(os.path.join(args.outdir, "allocation_statistics.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(alloc_rows[0].keys()))
                w.writeheader()
                w.writerows(alloc_rows)
            from collections import Counter
            det_c, da_c, lane_c = (Counter(r[k] for r in alloc_rows) for k in ("det", "da", "lane"))
            print("allocation stats (per-task budget index distribution):")
            print("  det :", dict(sorted(det_c.items())))
            print("  da  :", dict(sorted(da_c.items())))
            print("  lane:", dict(sorted(lane_c.items())))
        print(f"saved -> {args.outdir}/metrics.json")


if __name__ == "__main__":
    main()
