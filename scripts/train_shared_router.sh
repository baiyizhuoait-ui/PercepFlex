#!/usr/bin/env bash
# Train the shared-router variant (Exp2 Method A: one allocation for all tasks).
# Reuses the Stage A encoder; trains shared router + heads (Stage B) + joint (C) + budget (D).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
A_CKPT=experiments/exp_train_A/checkpoint.pt
$PY training/train.py --config configs/train_stageB_shared.yaml --init "$A_CKPT" --epochs 1 \
    --outdir experiments/exp_train_shared_B
$PY training/train.py --config configs/train_stageD_shared.yaml --init experiments/exp_train_shared_B/checkpoint.pt \
    --epochs 2 --outdir experiments/exp_train_shared_D
echo "DONE shared router training"
