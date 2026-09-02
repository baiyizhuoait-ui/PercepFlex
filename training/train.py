#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training pipeline (Phase 1 Step 10, taskbook §10 schedule).

Stages:
  A: train the fixed max-capacity model (StaticMultiTaskModel, no router)
  B: freeze encoder, train Router + heads (AdaptiveMultiTaskModel)
  C: joint fine-tune of the adaptive model
  D: budget-aware training (lambda_budget > 0, budget_target envelope)

Usage:
    gpu_env/bin/python training/train.py --config configs/train_stageA.yaml
    gpu_env/bin/python training/train.py --config configs/train_stageD.yaml --num-images 1000 --epochs 2
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datasets.bdd100k import BDD100KDataset, collate_train  # noqa: E402
from models.static_model import StaticMultiTaskModel  # noqa: E402
from models.adaptive_model import AdaptiveMultiTaskModel  # noqa: E402
from losses.multitask_loss import MultiTaskLoss  # noqa: E402

DEFAULT_ANCHORS = torch.tensor([[[4, 12], [7, 19], [11, 28]],
                                [[17, 40], [25, 58], [38, 89]],
                                [[62, 136], [88, 206], [124, 412]]])


def resolve_device(args):
    """Pick the device and print an explicit banner so a silent CPU fallback can
    never go unnoticed. Refuses to run the real pipeline on CPU unless the user
    opts in with --allow-cpu (intended only for tiny smoke tests)."""
    cuda_ok = torch.cuda.is_available()
    if args.device:
        dev_str = args.device.lower()
    else:
        dev_str = "cuda" if cuda_ok else "cpu"
    device = torch.device(dev_str)

    if dev_str.startswith("cuda") and not cuda_ok:
        sys.exit("[FATAL] --device cuda requested but torch has NO usable CUDA GPU.\n"
                 "        Check `nvidia-smi`, `python -c 'import torch;print(torch.__version__,"
                 " torch.version.cuda)'`\n"
                 "        and that a CPU-only torch wheel was not installed over the CUDA one.")

    if dev_str == "cpu" and not cuda_ok and not args.allow_cpu:
        sys.exit("[FATAL] CUDA GPU not detected and no explicit --device given.\n"
                 "        Training would silently run on CPU. To force CPU for a tiny\n"
                 "        smoke test only, rerun with: --device cpu --allow-cpu\n"
                 "        Otherwise fix the torch/CUDA environment first.")

    if cuda_ok:
        idx = device.index if device.type == "cuda" else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        cc = torch.cuda.get_device_capability(idx)
        print(f"\n=== GPU ENV =============================================", flush=True)
        print(f"  torch            : {torch.__version__}  (cuda build: {torch.version.cuda})", flush=True)
        print(f"  cuda.is_available: {cuda_ok}", flush=True)
        print(f"  device (used)    : {device}", flush=True)
        print(f"  GPU              : {name}  | {props.total_memory/1024**3:.0f} GiB | compute {cc}", flush=True)
        print(f"========================================================\n", flush=True)
    else:
        print(f"\n=== GPU ENV =============================================", flush=True)
        print(f"  torch            : {torch.__version__}  (cuda build: {torch.version.cuda})", flush=True)
        print(f"  cuda.is_available: {cuda_ok}", flush=True)
        print(f"  device (used)    : {device}   <-- CPU", flush=True)
        print(f"========================================================\n", flush=True)
    return device


def gpu_usage(idx=0):
    """Live GPU utilization + memory, robust across environments.
    Prefers torch.cuda.utilization; falls back to `nvidia-smi`; last resort = allocated memory."""
    parts = []
    try:
        util = torch.cuda.utilization(idx)
        parts.append(f"gpu% {int(util):3d}")
    except Exception:
        pass
    if not parts:  # nvidia-smi fallback (real utilization on the server)
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(idx)],
                capture_output=True, text=True, timeout=3)
            if out.returncode == 0:
                u, mu, mt = [x.strip() for x in out.stdout.split(",")]
                parts.append(f"gpu% {u:>3} mem {mu}/{mt}MiB")
        except Exception:
            pass
    if not parts:
        try:
            a = torch.cuda.memory_allocated(idx) / 1024**2
            r = torch.cuda.memory_reserved(idx) / 1024**2
            parts.append(f"mem {a:6.0f}/{r:6.0f}MiB")
        except Exception:
            parts.append("gpu?")
    return " | ".join(parts)


def build_model(cfg, adaptive):
    if adaptive:
        return AdaptiveMultiTaskModel(cfg["model"])
    return StaticMultiTaskModel(cfg["model"])


def train_one_epoch(model, loader, loss_fn, opt, device, cfg, stage, epoch,
                    log_path, max_steps=None, kd=None, kd_offline=None,
                    print_every=20):
    model.train()
    t0 = time.time()
    running = {}
    steps = 0
    for it, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        det_t = batch["det_targets"].to(device)
        da_m = batch["da_mask"].to(device)
        lane_m = batch["lane_mask"].to(device)
        opt.zero_grad()

        ref = None
        if stage == "A":
            if kd_offline is not None:
                det, da, lane, z_student = model(img, return_z=True)
            else:
                det, da, lane = model(img)
            routing = None
        elif cfg["train"].get("uniform_width"):
            # train heads at uniformly-sampled widths (all tasks, one width per batch)
            bi = torch.randint(0, len(model.budgets), ()).item()
            w = model.budgets[bi]
            fw = torch.tensor([w, w, w]).to(device)
            det, da, lane, routing = model(img, hard=True, return_routing=True, force_widths=fw)
        else:
            # budget prior per stage
            prior = None
            if cfg["train"].get("budget_mode") and cfg["train"]["budget_mode"] != "NONE":
                from models.router.task_resource_router import TaskResourceRouter
                prior = TaskResourceRouter.budget_prior_from_mode(
                    cfg["train"]["budget_mode"], len(model.budgets), device=device)
            diff_mode = cfg["model"].get("router", {}).get("type") == "difficulty"
            det, da, lane, routing, ref = model(img, budget_prior=prior, hard=False,
                                                soft=True, return_routing=True,
                                                return_ref=diff_mode)
            if not diff_mode:
                ref = None

        total, losses = loss_fn(det, da, lane, det_t, da_m, lane_m, routing,
                                img_size=cfg["model"].get("input_size", [640, 640])[0],
                                ref=ref, lambda_diff=float(cfg["train"].get("lambda_diff", 0.0)))
        if kd_offline is not None:
            # fetch cached teacher targets for this batch
            ids = [kd_offline["name2idx"][n] for n in batch["name"]]
            feat_t = torch.from_numpy(kd_offline["feat"][ids]).float().to(device)
            da_t = torch.from_numpy(kd_offline["da"][ids]).float().to(device)
            lane_t = torch.from_numpy(kd_offline["lane"][ids]).float().to(device)
            # z from the shared encoder+representation (compute here for both static & adaptive)
            feats_m = model.encoder(img)
            z_student = model.representation(feats_m)["z"]
            # feature alignment (upsample cache 1/16 -> 1/8)
            if feat_t.shape[2:] != z_student.shape[2:]:
                feat_t = torch.nn.functional.interpolate(
                    feat_t, size=z_student.shape[2:], mode="bilinear", align_corners=False)
            from distillation.kd import KDProjection
            if not hasattr(model, "_kd_proj"):
                model._kd_proj = KDProjection(feat_t.shape[1], 64).to(device)
            proj = model._kd_proj(feat_t)
            _kd = cfg["train"]["kd"]
            kd_losses = {
                "feat": torch.nn.functional.mse_loss(proj, z_student) * float(_kd.get("lambda_feat", 0.1)),
                "out_da": torch.nn.functional.mse_loss(
                    torch.nn.functional.interpolate(torch.sigmoid(da), size=da_t.shape[2:],
                                                    mode="bilinear", align_corners=False),
                    torch.sigmoid(da_t)) * float(_kd.get("lambda_out", 1.0)),
                "out_lane": torch.nn.functional.mse_loss(
                    torch.nn.functional.interpolate(torch.sigmoid(lane), size=lane_t.shape[2:],
                                                    mode="bilinear", align_corners=False),
                    torch.sigmoid(lane_t)) * float(_kd.get("lambda_out", 1.0)),
            }
            for k, v in kd_losses.items():
                total = total + v
                losses[k] = v
            losses["total"] = total
        elif kd is not None:
            z_student = None
            if hasattr(model, "representation"):
                feats_m = model.encoder(img)
                z_student = model.representation(feats_m)["z"]
            kd_losses = kd(img, z_student, da, lane)
            for k, v in kd_losses.items():
                total = total + v
                losses[k] = v
            losses["total"] = total
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        steps += 1
        if (it + 1) % print_every == 0 or it == 0:
            avg = {k: v / steps for k, v in running.items()}
            el = time.time() - t0
            done = it + 1
            ipm = done / el if el > 0 else 0.0
            eta_s = (len(loader) - done) / ipm if ipm > 0 else 0.0
            gpu = ""
            if device.type == "cuda":
                idx = device.index if device.index is not None else 0
                gpu = gpu_usage(idx)
            lr = opt.param_groups[0]["lr"]
            line = (f"[{time.strftime('%H:%M:%S')}] st{stage} ep{epoch} "
                    f"step {done}/{len(loader)} | {el/60:.1f}m in, ETA {eta_s/60:.1f}m "
                    f"| {el/done*1e3:.0f}ms/step | lr {lr:.2e} | {gpu}"
                    f" | loss {avg['total']:.4f} det {avg.get('det',0):.4f} "
                    f"da {avg.get('da',0):.4f} lane {avg.get('lane',0):.4f} "
                    f"budget {avg.get('budget',0):.4f}"
                    + ' '.join(f" {k} {avg.get(k,0):.3f}" for k in ('feat', 'out_da', 'out_lane') if k in avg))
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        if max_steps and steps >= max_steps:
            break
    return {k: v / steps for k, v in running.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-images", type=int, default=0, help="0 = full split")
    ap.add_argument("--epochs", type=int, default=0, help="override epochs")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--init", default=None, help="checkpoint to initialize weights from")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default=None, help="cuda[:N] or cpu (auto: cuda if available)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit running on CPU (tiny smoke tests only)")
    ap.add_argument("--log-every", type=int, default=20,
                    help="print progress every N steps (0 = only epoch summary)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader workers (default: auto ~ cores; low values starve the GPU)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)
    import random as _rnd
    _rnd.seed(args.seed)
    stage = cfg["train"]["stage"]
    adaptive = stage in ("B", "C", "D")
    outdir = args.outdir or cfg["train"].get("outdir",
                                             os.path.join(ROOT, "experiments", f"exp_train_{stage}"))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    log_path = os.path.join(outdir, "training_log.txt")

    device = resolve_device(args)
    model = build_model(cfg, adaptive).to(device)

    # ---- KD teacher (optional) ----
    kd_cfg = cfg["train"].get("kd")
    kd = None
    kd_offline = None
    if kd_cfg and kd_cfg.get("offline_cache"):
        import numpy as _np
        from datasets.bdd100k import BDD100KDataset as _DS
        cache_dir = os.path.join(ROOT, kd_cfg["offline_cache"])
        tname = kd_cfg["teacher"]
        manifest = open(os.path.join(cache_dir, f"{tname}_manifest.txt")).read().split()
        selfeat = _np.load(os.path.join(cache_dir, f"{tname}_feat.npy"), mmap_mode="r")
        selfda = _np.load(os.path.join(cache_dir, f"{tname}_da.npy"), mmap_mode="r")
        selflane = _np.load(os.path.join(cache_dir, f"{tname}_lane.npy"), mmap_mode="r")
        name2idx = {n: i for i, n in enumerate(manifest)}
        kd_offline = {"names": manifest, "name2idx": name2idx,
                      "feat": selfeat, "da": selfda, "lane": selflane}
        print(f"[train] OFFLINE KD teacher={tname} cache={cache_dir} "
              f"({len(manifest)} imgs)", flush=True)
    if kd_cfg and not kd_offline:
        from distillation.kd import CrossArchKD
        tname = kd_cfg["teacher"]
        if tname == "YOLOP":
            from evaluation.evaluate_baseline import build_yolop
            teacher = build_yolop(device)
            norm = "imagenet"
        else:
            from load_baseline_weights import load_baseline
            preset = kd_cfg.get("preset")
            teacher = load_baseline(tname, preset, device)
            norm = "imagenet" if tname == "TriLiteNet" else "unit"
        kd = CrossArchKD(tname, teacher, 64,
                         lambda_feat=float(kd_cfg.get("lambda_feat", 0.1)),
                         lambda_out=float(kd_cfg.get("lambda_out", 1.0)),
                         T=float(kd_cfg.get("T", 4.0)), norm=norm).to(device)
        print(f"[train] KD teacher={tname} lambda_feat={kd_cfg.get('lambda_feat',0.1)} "
              f"lambda_out={kd_cfg.get('lambda_out',1.0)}", flush=True)
    if cfg["train"].get("static_width"):
        w = float(cfg["train"]["static_width"])
        model.set_width(w)
        print(f"[train] static model at fixed width {w}", flush=True)
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        sd = ckpt.get("model_state", ckpt)
        # keep only keys with matching shapes (skip e.g. a different router type)
        cur = model.state_dict()
        compat = {k: v for k, v in sd.items()
                  if k in cur and cur[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(compat, strict=False)
        print(f"[train] init from {args.init}: loaded {len(compat)}/{len(sd)} keys "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] stage={stage} adaptive={adaptive} params={n_params/1e6:.3f}M", flush=True)

    tr = cfg["train"]
    anchors = torch.tensor(tr.get("anchors", DEFAULT_ANCHORS.tolist()), dtype=torch.float32)
    loss_fn = MultiTaskLoss(
        anchors, nc=cfg["model"].get("detection", {}).get("nc", 1),
        lambda_da=tr.get("lambda_da", 1.0), lambda_lane=tr.get("lambda_lane", 1.0),
        lambda_budget=tr.get("lambda_budget", 0.0),
        budget_target=tr.get("budget_target"),
        width_penalty=tr.get("width_penalty", 1.0)).to(device)

    ds = BDD100KDataset(cfg["data"]["root"], split=tr.get("train_split", "tri_train"),
                        train=True)
    if args.num_images:
        ds.names = ds.names[:args.num_images]
        if getattr(ds, "_det_by_name", None):
            keep = set(ds.names)
            ds._det_by_name = {k: v for k, v in ds._det_by_name.items() if k in keep}
    g = torch.Generator()
    g.manual_seed(args.seed)
    # --- DataLoader workers: the main lever for GPU utilization ---
    # On a real multi-core box, num_workers=2 starves the GPU (gpu% in the low
    # single digits) because most wall-time is spent on CPU preprocessing/disk
    # reads. Auto-raise it when the config default (<=2) is clearly too low.
    cfg_nw = tr.get("num_workers", 2)
    nw = args.num_workers or cfg_nw
    if args.num_workers is None and cfg_nw <= 2 and device.type == "cuda":
        auto_nw = min(8, max(4, (os.cpu_count() or 8) // 2))
        if auto_nw > cfg_nw:
            nw = auto_nw
    pin = device.type == "cuda"
    pf = max(2, int(tr.get("prefetch_factor", 4)))
    loader = DataLoader(ds, batch_size=tr.get("batch_size", 8), shuffle=True,
                        num_workers=nw, collate_fn=collate_train, drop_last=True,
                        generator=g, pin_memory=pin,
                        prefetch_factor=(pf if nw > 0 else None))
    if nw != cfg_nw:
        print(f"[train] DataLoader workers {cfg_nw} -> {nw} (auto-raised; "
              f"pass --num-workers to override) pin_memory={pin} prefetch={pf}", flush=True)

    epochs = args.epochs or int(tr.get("epochs", 1))
    lr = float(tr.get("lr", 1e-3))
    if stage == "B" or tr.get("freeze_encoder"):
        # freeze encoder (+ representation)
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.representation.parameters():
            p.requires_grad = False
    if tr.get("freeze_heads"):
        # freeze task heads — only the router trains
        for h in (model.det_head, model.da_head, model.lane_head):
            for p in h.parameters():
                p.requires_grad = False
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                      weight_decay=float(tr.get("weight_decay", 5e-4)))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(len(loader), 1))

    for ep in range(1, epochs + 1):
        avg = train_one_epoch(model, loader, loss_fn, opt, device, cfg, stage, ep,
                              log_path, max_steps=tr.get("max_steps"), kd=kd,
                              kd_offline=kd_offline,
                              print_every=max(args.log_every, 1))
        sched.step()
        ckpt = os.path.join(outdir, "checkpoint.pt")
        torch.save({"epoch": ep, "model_state": model.state_dict(),
                    "opt_state": opt.state_dict()}, ckpt)
        line = f"[stage {stage}] ep {ep} DONE avg_loss={avg.get('total',0):.4f} saved {ckpt}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"stage": stage, "params": n_params, "epochs": epochs,
                   "final_avg_loss": avg.get("total", 0)}, f, indent=2)
    print(f"[train] done -> {outdir}")


if __name__ == "__main__":
    main()
