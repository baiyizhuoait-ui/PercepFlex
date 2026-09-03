#!/bin/bash
# Phase-2-B pre-flight smoke test.
# Verifies (1) R0 still trains after the flag-gated static_model.py edit,
# (2) the new R2 det-from-Z path runs, (3) the train->eval wiring (config.yaml
# next to checkpoint) restores the correct z_channels / from_z architecture.
# Tiny: 1 epoch, 160 train images, 60 eval images.
set -x
PY=/home/mycode/ai_study/gpu_env/bin/python
cd /home/mycode/ai_study/trac || exit 1
S=experiments/phase2b/smoke
mkdir -p "$S"

echo "===== SMOKE 1/4: R0 z=16 train ====="
"$PY" training/train.py --config configs/phase2b_zsweep_z16.yaml \
  --outdir "$S/r0z16" --epochs 1 --num-images 160 --seed 0

echo "===== SMOKE 2/4: R0 z=16 eval ====="
"$PY" evaluation/evaluate_baseline.py --baseline OursStatic \
  --preset "$S/r0z16/checkpoint.pt" --outdir "$S/r0z16_eval" --num-images 60

echo "===== SMOKE 3/4: R2 z=32 train (det from Z) ====="
"$PY" training/train.py --config configs/phase2b_r2_zsweep_z32.yaml \
  --outdir "$S/r2z32" --epochs 1 --num-images 160 --seed 0

echo "===== SMOKE 4/4: R2 z=32 eval ====="
"$PY" evaluation/evaluate_baseline.py --baseline OursStatic \
  --preset "$S/r2z32/checkpoint.pt" --outdir "$S/r2z32_eval" --num-images 60

echo "===== SMOKE_DONE ====="
