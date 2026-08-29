#!/usr/bin/env bash
# Experiment I: KD 后重新测试 Dynamic.
# A = Static + No KD   (exp_train_A / static_eb)
# B = Dynamic + No KD   (exp_train_D)
# C = Static + CrossArch KD (expF kd_yolop student)
# D = Dynamic + CrossArch KD (KD student -> B/C/D pipeline with KD)
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expI_kd_dynamic
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }

# C: static + KD = the best single-teacher student from F/G (evaluate full val)
log "C: static+KD eval (best teacher student)"
BEST_TEACHER=YOLOP
$PY evaluation/evaluate_baseline.py --baseline OursStatic \
    --preset "experiments/expF_single_teacher/kd_$(echo $BEST_TEACHER | tr 'A-Z' 'a-z')/checkpoint.pt" \
    --outdir "$OUT/static_kd_eval" > "$OUT/static_kd_eval.log" 2>&1
log "  C done"

# D: KD student -> dynamic pipeline (B/C/D stages with KD on)
log "D: KD-dynamic pipeline"
$PY training/train.py --config configs/train_stageB_kd.yaml \
    --init "experiments/expF_single_teacher/kd_yolop/checkpoint.pt" --epochs 2 \
    --outdir "$OUT/kdD_B" > "$OUT/kdD_B.log" 2>&1 || { log "  kdD_B FAIL"; tail -3 "$OUT/kdD_B.log"; exit 1; }
sed 's/stage: B/stage: C/; s/lr: 3e-4/lr: 1e-4/' configs/train_stageB_kd.yaml > "$OUT/stageC_kd.yaml"
$PY training/train.py --config "$OUT/stageC_kd.yaml" \
    --init "$OUT/kdD_B/checkpoint.pt" --epochs 2 \
    --outdir "$OUT/kdD_C" > "$OUT/kdD_C.log" 2>&1 || { log "  kdD_C FAIL"; tail -3 "$OUT/kdD_C.log"; exit 1; }
$PY training/train.py --config configs/train_stageD.yaml \
    --init "$OUT/kdD_C/checkpoint.pt" --epochs 2 \
    --outdir "$OUT/kdD_D" > "$OUT/kdD_D.log" 2>&1 || { log "  kdD_D FAIL"; tail -3 "$OUT/kdD_D.log"; exit 1; }

log "D eval (dynamic+KD)"
$PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$OUT/kdD_D/checkpoint.pt" \
    --outdir "$OUT/kdD_eval" --alloc-stats > "$OUT/kdD_eval.log" 2>&1
log "EXP-I DONE"
