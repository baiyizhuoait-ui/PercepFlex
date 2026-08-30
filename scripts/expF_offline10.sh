#!/usr/bin/env bash
# F/G offline KD with 10 epochs (fairer screening).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expF_single_teacher
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }
train_offkd() {
  local tag="$1" teacher="$2"
  log "offline-KD train x10 (teacher=$teacher)"
  $PY training/train.py --config "configs/train_stageA_offkd_${teacher}.yaml" \
      --num-images 10000 --epochs 10 --outdir "$OUT/${tag}" > "$OUT/${tag}.log" 2>&1
  [ -f "$OUT/$tag/checkpoint.pt" ] || { log "  $tag TRAIN FAIL"; tail -5 "$OUT/$tag.log"; return 1; }
  log "eval $tag"
  $PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$OUT/$tag/checkpoint.pt" \
      --outdir "$OUT/${tag}_eval" > "$OUT/${tag}_eval.log" 2>&1
  log "  $tag done"
}
train_offkd kd_yolop_off10 YOLOP
train_offkd kd_twinlite_off10 TwinLiteNetPlus
train_offkd kd_trilite_off10 TriLiteNet
log "EXP-F/G-OFF10 DONE"
