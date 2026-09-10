#!/usr/bin/env bash
# Train the shared-router variant (Exp2 Method A: one allocation for all tasks).
# Reuses the Stage A encoder; trains shared router + heads (Stage B) + joint (C) + budget (D).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
A_CKPT=experiments/phase1b/exp_train_a/checkpoint.pt
$PY training/train.py --config configs/phase1b_train_stage_b_shared.yaml --init "$A_CKPT" --epochs 1 \
    --outdir experiments/phase1b/exp_train_shared_b
$PY training/train.py --config configs/phase1b_train_stage_d_shared.yaml --init experiments/phase1b/exp_train_shared_b/checkpoint.pt \
    --epochs 2 --outdir experiments/phase1b/exp_train_shared_d
echo "DONE shared router training"
