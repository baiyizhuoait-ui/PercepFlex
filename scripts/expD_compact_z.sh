#!/usr/bin/env bash
# Experiment D: Compact Representation at equal-parameter budgets (~0.5/1.0/2.0M).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expD_compact_z
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/status.txt"; }

for tag in 0.5M 1.0M 2.0M; do
  log "train ours_$tag"
  $PY training/train.py --config "configs/train_stageA_${tag}.yaml" \
      --outdir "$OUT/ours_$tag" > "$OUT/ours_${tag}.log" 2>&1
  [ -f "$OUT/ours_$tag/checkpoint.pt" ] || { log "  ours_$tag FAIL"; tail -3 "$OUT/ours_${tag}.log"; continue; }
  log "eval ours_$tag"
  $PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$OUT/ours_$tag/checkpoint.pt" \
      --outdir "$OUT/ours_${tag}_eval" > "$OUT/ours_${tag}_eval.log" 2>&1
  log "  ours_$tag done"
done
log "EXP-D DONE"
