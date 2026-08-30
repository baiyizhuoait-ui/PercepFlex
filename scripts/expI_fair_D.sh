#!/usr/bin/env bash
# Fair D: init from full YOLOP-KD student, dynamic pipeline on FULL 70k (no per-step KD;
# the KD improvement is baked into the init representation).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expI_kd_dynamic
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }
log "fair D: full-70k dynamic from KD init"
$PY training/train.py --config configs/train_stageB.yaml \
    --init "experiments/expF_single_teacher/kd_yolop/checkpoint.pt" --epochs 2 \
    --outdir "$OUT/fairD_B" > "$OUT/fairD_B.log" 2>&1 || { log " fairD_B FAIL"; tail -3 "$OUT/fairD_B.log"; exit 1; }
$PY training/train.py --config configs/train_stageC.yaml \
    --init "$OUT/fairD_B/checkpoint.pt" --epochs 2 \
    --outdir "$OUT/fairD_C" > "$OUT/fairD_C.log" 2>&1 || { log " fairD_C FAIL"; tail -3 "$OUT/fairD_C.log"; exit 1; }
$PY training/train.py --config configs/train_stageD.yaml \
    --init "$OUT/fairD_C/checkpoint.pt" --epochs 2 \
    --outdir "$OUT/fairD_D" > "$OUT/fairD_D.log" 2>&1 || { log " fairD_D FAIL"; tail -3 "$OUT/fairD_D.log"; exit 1; }
log "fair D eval"
$PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$OUT/fairD_D/checkpoint.pt" \
    --outdir "$OUT/fairD_eval" --alloc-stats > "$OUT/fairD_eval.log" 2>&1
log "FAIR-D DONE"
