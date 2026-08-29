#!/usr/bin/env bash
# Experiment A: Equal-Budget Static vs Dynamic — 3 seeds.
# static_eb: independently trained static model at width 0.45 (~0.90G, = dynamic avg)
# dynamic:   A->B->C->D pipeline (seed 0 reuses exp_train_D; seeds 1-2 retrained)
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expA_equal_budget
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }

for seed in 0 1 2; do
  log "static_eb seed=$seed"
  if [ "$seed" = "0" ] && [ -d "$OUT/static_eb_s0" ]; then
    log "  static_eb_s0 already exists (training)"
  else
    $PY training/train.py --config configs/train_stageA_eb.yaml --seed "$seed" \
        --outdir "$OUT/static_eb_s$seed" > "$OUT/static_eb_s$seed.log" 2>&1 || { log "  static_eb_s$seed FAIL"; continue; }
  fi
  log "dynamic seed=$seed"
  if [ "$seed" = "0" ] && [ -d experiments/exp_train_D ]; then
    log "  dynamic s0 = exp_train_D (existing)"
  else
    # dynamic pipeline A->B->C->D with seed
    $PY training/train.py --config configs/train_stageA.yaml --seed "$seed" \
        --epochs 4 --outdir "$OUT/dynA_s$seed" > "$OUT/dynA_s$seed.log" 2>&1 || { log "  dynA_s$seed FAIL"; continue; }
    $PY training/train.py --config configs/train_stageB.yaml --seed "$seed" \
        --init "$OUT/dynA_s$seed/checkpoint.pt" --epochs 1 \
        --outdir "$OUT/dynB_s$seed" > "$OUT/dynB_s$seed.log" 2>&1 || continue
    $PY training/train.py --config configs/train_stageC.yaml --seed "$seed" \
        --init "$OUT/dynB_s$seed/checkpoint.pt" --epochs 1 \
        --outdir "$OUT/dynC_s$seed" > "$OUT/dynC_s$seed.log" 2>&1 || continue
    $PY training/train.py --config configs/train_stageD.yaml --seed "$seed" \
        --init "$OUT/dynC_s$seed/checkpoint.pt" --epochs 2 \
        --outdir "$OUT/dynD_s$seed" > "$OUT/dynD_s$seed.log" 2>&1 || continue
  fi
done
log "EXP-A TRAINING DONE"
