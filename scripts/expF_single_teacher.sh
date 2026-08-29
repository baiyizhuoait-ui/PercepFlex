#!/usr/bin/env bash
# Experiment F/G: Single-Teacher Cross-Architecture KD.
# Baseline = exp_train_A (No KD). Train student with each teacher, eval on full val.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expF_single_teacher
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }

train_kd() {
  local tag="$1" teacher="$2"
  log "train KD student (teacher=$teacher)"
  $PY training/train.py --config "configs/train_stageA_kd_${teacher}.yaml" \
      --outdir "$OUT/${tag}" > "$OUT/${tag}.log" 2>&1
  [ -f "$OUT/$tag/checkpoint.pt" ] || { log "  $tag TRAIN FAIL"; tail -3 "$OUT/$tag.log"; return 1; }
  log "eval $tag"
  $PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$OUT/$tag/checkpoint.pt" \
      --outdir "$OUT/${tag}_eval" > "$OUT/${tag}_eval.log" 2>&1
  log "  $tag done"
}

# no-KD baseline already exists as exp_train_A; copy its eval reference
train_kd kd_yolop YOLOP
train_kd kd_twinlite TwinLiteNetPlus
train_kd kd_trilite TriLiteNet
log "EXP-F/G DONE"
