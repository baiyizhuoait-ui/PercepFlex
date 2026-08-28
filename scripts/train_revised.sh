#!/usr/bin/env bash
# Revised pipeline: A -> U (uniform-width heads) -> C (router on frozen heads) -> D (joint soft)
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
EXP=experiments/revised
mkdir -p "$EXP"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$EXP/status.txt"; }

log "STAGE U (uniform-width heads)"
$PY training/train.py --config configs/train_stageU.yaml --init experiments/exp_train_A/checkpoint.pt \
    --epochs 2 --outdir "$EXP/exp_U" > "$EXP/exp_U.log" 2>&1
[ -f "$EXP/exp_U/checkpoint.pt" ] && log "U done" || { log "U FAIL"; tail -3 "$EXP/exp_U.log"; exit 1; }

log "STAGE C (router on frozen heads, soft)"
$PY training/train.py --config configs/train_stageC_router.yaml --init "$EXP/exp_U/checkpoint.pt" \
    --epochs 2 --outdir "$EXP/exp_C" > "$EXP/exp_C.log" 2>&1
[ -f "$EXP/exp_C/checkpoint.pt" ] && log "C done" || { log "C FAIL"; tail -3 "$EXP/exp_C.log"; exit 1; }

log "STAGE D (joint soft, budget-aware)"
$PY training/train.py --config configs/train_stageD.yaml --init "$EXP/exp_C/checkpoint.pt" \
    --epochs 2 --outdir "$EXP/exp_D" > "$EXP/exp_D.log" 2>&1
[ -f "$EXP/exp_D/checkpoint.pt" ] && log "D done" || { log "D FAIL"; tail -3 "$EXP/exp_D.log"; exit 1; }
log "REVISED PIPELINE COMPLETE"
