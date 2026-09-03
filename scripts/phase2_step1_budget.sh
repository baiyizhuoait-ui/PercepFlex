#!/usr/bin/env bash
# Phase-2 Step 1: fixed-protocol equal-budget baselines (0.5/1.0/2.0M), serial on one GPU.
set -u
cd "$(dirname "$0")/.."
PY=/home/mycode/ai_study/gpu_env/bin/python
OUT=experiments/phase2
st(){ echo "$(date +%H:%M:%S) $*" >> "$OUT/status.txt"; }

for cfg in 0.5M 1.0M 2.0M; do
  tag="A_budget_${cfg}"
  if [ -f "$OUT/$tag/checkpoint.pt" ]; then st "$tag exists, skip"; continue; fi
  st "START $tag"
  $PY training/train.py --config "configs/train_stageA_${cfg}.yaml" --epochs 4 --seed 0 \
      --outdir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  if [ -f "$OUT/$tag/checkpoint.pt" ]; then st "DONE $tag"; else st "FAIL $tag"; fi
done
st "STEP1_BUDGET_CHAIN_DONE"
