#!/usr/bin/env bash
# Complete Experiment A: finish s2 pipeline, run all evals, produce report.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expA_equal_budget
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/finish.log"; }

# wait for dynA_s2 to finish (no train.py process on it)
while pgrep -f "dynA_s2" > /dev/null 2>&1; do sleep 60; done
[ -f "$OUT/dynA_s2/checkpoint.pt" ] || { log "dynA_s2 missing"; exit 1; }
log "dynA_s2 done"

for s in 2; do
  log "s$s B"
  $PY training/train.py --config configs/train_stageB.yaml --seed $s --init "$OUT/dynA_s$s/checkpoint.pt" \
      --epochs 2 --outdir "$OUT/dynB_s$s" > "$OUT/dynB_s$s.log" 2>&1 || { log "s$s B FAIL"; exit 1; }
  log "s$s C"
  $PY training/train.py --config configs/train_stageC.yaml --seed $s --init "$OUT/dynB_s$s/checkpoint.pt" \
      --epochs 2 --outdir "$OUT/dynC_s$s" > "$OUT/dynC_s$s.log" 2>&1 || { log "s$s C FAIL"; exit 1; }
  log "s$s D"
  $PY training/train.py --config configs/train_stageD.yaml --seed $s --init "$OUT/dynC_s$s/checkpoint.pt" \
      --epochs 2 --outdir "$OUT/dynD_s$s" > "$OUT/dynD_s$s.log" 2>&1 || { log "s$s D FAIL"; exit 1; }
done
log "s2 pipeline done"

# evals
for s in 1 2; do
  log "eval dynamic s$s"
  $PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$OUT/dynD_s$s/checkpoint.pt" \
      --outdir "$OUT/dynD_s${s}_eval" --alloc-stats > "$OUT/dynD_s${s}_eval.log" 2>&1
done
log "eval static s1 lane fix"
$PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$OUT/static_eb_s1/checkpoint.pt" \
    --static-width 0.45 --outdir "$OUT/static_eb_s1_eval" > "$OUT/static_eb_s1_eval.log" 2>&1

log "=== ExpA analysis ==="
$PY scripts/expA_analyze.py "$OUT"
log "EXP-A COMPLETE"
