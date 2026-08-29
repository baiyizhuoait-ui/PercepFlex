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

from datasets.bdd100k import BDD100KDataset, collate_train  # noqa: E402
from models.static_model import StaticMultiTaskModel  # noqa: E402
from models.adaptive_model import AdaptiveMultiTaskModel  # noqa: E402
from losses.multitask_loss import MultiTaskLoss  # noqa: E402

DEFAULT_ANCHORS = torch.tensor([[[4, 12], [7, 19], [11, 28]],
                                [[17, 40], [25, 58], [38, 89]],
                                [[62, 136], [88, 206], [124, 412]]])


def build_model(cfg, adaptive):
    if adaptive:
        return AdaptiveMultiTaskModel(cfg["model"])
    return StaticMultiTaskModel(cfg["model"])


def train_one_epoch(model, loader, loss_fn, opt, device, cfg, stage, epoch,
                    log_path, max_steps=None):
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
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v)
        steps += 1
        if (it + 1) % 50 == 0:
            avg = {k: v / steps for k, v in running.items()}
            line = (f"[stage {stage}] ep {epoch} step {it+1}/{len(loader)} "
                    f"loss {avg['total']:.4f} det {avg.get('det',0):.4f} "
                    f"da {avg.get('da',0):.4f} lane {avg.get('lane',0):.4f} "
                    f"budget {avg.get('budget',0):.4f} "
                    f"{(time.time()-t0)/(it+1)*1e3:.0f}ms/step")
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    device = torch.device(args.device)
    model = build_model(cfg, adaptive).to(device)
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
    loader = DataLoader(ds, batch_size=tr.get("batch_size", 8), shuffle=True,
                        num_workers=tr.get("num_workers", 2), collate_fn=collate_train,
                        drop_last=True, generator=g)

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
                              log_path, max_steps=tr.get("max_steps"))
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
