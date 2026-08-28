#!/usr/bin/env bash
# Experiment 1 (taskbook §11): Static Tiny/Medium/Large vs Dynamic.
# Same trained adaptive model (exp_train_D/checkpoint.pt), allocation is the
# only variable: forced uniform widths vs free per-frame task-wise routing.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
EXP=experiments/exp_001
CKPT=${1:-experiments/exp_train_D/checkpoint.pt}
OUT=experiments/exp_static_vs_dynamic
mkdir -p "$OUT"
: > "$OUT/status.txt"

run() {
  local tag="$1"; shift
  echo "$(date +%H:%M:%S) START $tag" >> "$OUT/status.txt"
  $PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$CKPT" "$@" \
      --outdir "$OUT/$tag" --alloc-stats > "$OUT/$tag.log" 2>&1
  [ -f "$OUT/$tag/metrics.json" ] && echo "$(date +%H:%M:%S) DONE $tag" >> "$OUT/status.txt" \
      || echo "$(date +%H:%M:%S) FAIL $tag" >> "$OUT/status.txt"
}

echo "=== Experiment 1: Static vs Dynamic (ckpt=$CKPT) ===" | tee "$OUT/status.txt"
run static_tiny    --force-widths 0.25,0.25,0.25
run static_medium  --force-widths 0.5,0.5,0.5
run static_large   --force-widths 1.0,1.0,1.0
run dynamic
echo "$(date +%H:%M:%S) ALL_DONE" >> "$OUT/status.txt"
echo "=== summary ==="
$PY evaluation/summarize.py "$OUT"
