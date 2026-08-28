#!/usr/bin/env bash
# Quick Stage A validation on a 10k-image subset (fast iteration before full run).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
$PY training/train.py --config configs/train_stageA.yaml --num-images 10000 \
    --epochs 2 --outdir experiments/exp_train_A_subset
echo "=== quick eval (500 val images) ==="
$PY evaluation/evaluate_baseline.py --baseline OursStatic --preset experiments/exp_train_A_subset/checkpoint.pt \
    --num-images 500 --outdir experiments/exp_train_A_subset/eval_quick
