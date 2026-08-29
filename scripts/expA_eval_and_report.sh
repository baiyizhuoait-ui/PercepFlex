#!/usr/bin/env bash
# After all static_eb seeds train: evaluate each on full val, then run expA_analyze.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/expA_equal_budget
for s in 0 1 2; do
  [ -f "$OUT/static_eb_s$s/checkpoint.pt" ] || { echo "s$s missing checkpoint"; continue; }
  [ -f "$OUT/static_eb_s${s}_eval/metrics.json" ] && { echo "s$s eval exists"; continue; }
  echo "$(date +%H:%M:%S) eval s$s"
  $PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$OUT/static_eb_s$s/checkpoint.pt" \
      --static-width 0.45 --outdir "$OUT/static_eb_s${s}_eval" > "$OUT/static_eb_s${s}_eval.log" 2>&1
done
echo "=== ExpA analysis ==="
$PY scripts/expA_analyze.py "$OUT"
