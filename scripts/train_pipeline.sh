#!/usr/bin/env bash
# Phase 1 training pipeline: Stage A (static max) -> B (router+heads) -> C (joint) -> D (budget-aware)
# Each stage trains on tri_train and saves to experiments/exp_train_<stage>/.
# Stage A checkpoint initializes the adaptive model for B/C/D.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
EXP=experiments
mkdir -p "$EXP"
LOG="$EXP/pipeline_status.txt"
: > "$LOG"

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

# args: num_images (0=full), epochs per stage
IMAGES=${1:-0}
EPOCHS_A=${2:-1}
EPOCHS_B=${3:-1}
EPOCHS_C=${4:-1}
EPOCHS_D=${5:-1}

# ---- Stage A: static max-capacity ----
log "STAGE A start (static, epochs=$EPOCHS_A)"
$PY training/train.py --config configs/train_stageA.yaml --num-images "$IMAGES" --epochs "$EPOCHS_A" \
    --outdir "$EXP/exp_train_A" > "$EXP/exp_train_A.log" 2>&1
[ -f "$EXP/exp_train_A/checkpoint.pt" ] && log "STAGE A done" || { log "STAGE A FAILED"; tail -5 "$EXP/exp_train_A.log"; exit 1; }

# ---- Stage B: freeze encoder, train router + heads (init from A) ----
log "STAGE B start (router+heads)"
$PY training/train.py --config configs/train_stageB.yaml --num-images "$IMAGES" --epochs "$EPOCHS_B" \
    --init "$EXP/exp_train_A/checkpoint.pt" --outdir "$EXP/exp_train_B" > "$EXP/exp_train_B.log" 2>&1
[ -f "$EXP/exp_train_B/checkpoint.pt" ] && log "STAGE B done" || { log "STAGE B FAILED"; tail -5 "$EXP/exp_train_B.log"; exit 1; }

# ---- Stage C: joint fine-tune ----
log "STAGE C start (joint)"
$PY training/train.py --config configs/train_stageC.yaml --num-images "$IMAGES" --epochs "$EPOCHS_C" \
    --init "$EXP/exp_train_B/checkpoint.pt" --outdir "$EXP/exp_train_C" > "$EXP/exp_train_C.log" 2>&1
[ -f "$EXP/exp_train_C/checkpoint.pt" ] && log "STAGE C done" || { log "STAGE C FAILED"; tail -5 "$EXP/exp_train_C.log"; exit 1; }

# ---- Stage D: budget-aware ----
log "STAGE D start (budget-aware)"
$PY training/train.py --config configs/train_stageD.yaml --num-images "$IMAGES" --epochs "$EPOCHS_D" \
    --init "$EXP/exp_train_C/checkpoint.pt" --outdir "$EXP/exp_train_D" > "$EXP/exp_train_D.log" 2>&1
[ -f "$EXP/exp_train_D/checkpoint.pt" ] && log "STAGE D done" || { log "STAGE D FAILED"; tail -5 "$EXP/exp_train_D.log"; exit 1; }

log "PIPELINE COMPLETE"
