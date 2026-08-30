#!/usr/bin/env bash
# Experiment I: KD 后重新测试 Dynamic.
# A = Static no-KD (exp_train_A)  B = Dynamic no-KD (exp_train_D)
# C = Static+KD (best teacher, full online)  D = Dynamic+KD (offline KD pipeline)
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expI_kd_dynamic
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }

# D: KD student -> dynamic pipeline (B/C/D with offline KD)
log "D: KD-dynamic pipeline (init from full online YOLOP-KD student)"
$PY training/train.py --config configs/train_stageB_offkd.yaml \
    --init "experiments/expF_single_teacher/kd_yolop/checkpoint.pt" --epochs 2 \
    --num-images 10000 --outdir "$OUT/kdD_B" > "$OUT/kdD_B.log" 2>&1 || { log "  kdD_B FAIL"; tail -3 "$OUT/kdD_B.log"; exit 1; }
log "  kdD_B done"
$PY training/train.py --config configs/train_stageC_offkd.yaml \
    --init "$OUT/kdD_B/checkpoint.pt" --epochs 2 \
    --num-images 10000 --outdir "$OUT/kdD_C" > "$OUT/kdD_C.log" 2>&1 || { log "  kdD_C FAIL"; tail -3 "$OUT/kdD_C.log"; exit 1; }
log "  kdD_C done"
$PY training/train.py --config configs/train_stageD_offkd.yaml \
    --init "$OUT/kdD_C/checkpoint.pt" --epochs 2 \
    --num-images 10000 --outdir "$OUT/kdD_D" > "$OUT/kdD_D.log" 2>&1 || { log "  kdD_D FAIL"; tail -3 "$OUT/kdD_D.log"; exit 1; }
log "  kdD_D done"

log "D eval (dynamic+KD)"
$PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$OUT/kdD_D/checkpoint.pt" \
    --outdir "$OUT/kdD_eval" --alloc-stats > "$OUT/kdD_eval.log" 2>&1
log "EXP-I DONE"
