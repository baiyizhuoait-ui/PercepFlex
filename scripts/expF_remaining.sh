#!/usr/bin/env bash
# Run the remaining F/G teachers (twinlite, trilite) + then launch Exp I via master chain.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expF_single_teacher
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }
train_kd() {
  local tag="$1" teacher="$2"
  log "train KD student (teacher=$teacher)"
  $PY training/train.py --config "configs/train_stageA_kd_${teacher}.yaml" \
      --outdir "$OUT/${tag}" > "$OUT/${tag}.log" 2>&1
  [ -f "$OUT/$tag/checkpoint.pt" ] || { log "  $tag TRAIN FAIL"; tail -5 "$OUT/$tag.log"; return 1; }
  log "eval $tag"
  $PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$OUT/$tag/checkpoint.pt" \
      --outdir "$OUT/${tag}_eval" > "$OUT/${tag}_eval.log" 2>&1
  log "  $tag done"
}
train_kd kd_twinlite TwinLiteNetPlus
train_kd kd_trilite TriLiteNet
log "EXP-F/G DONE"
echo "=== F/G analysis ==="
$PY scripts/expFG_analyze.py | tee "$OUT/analysis.txt"
log "launching Exp I"
./scripts/expI_kd_dynamic.sh
log "PHASE1B CHAIN COMPLETE"
