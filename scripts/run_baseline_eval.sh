#!/usr/bin/env bash
# Phase 1 Step 3: full unified baseline evaluation on tri_val (10,000 images).
# One log file per model (no pipes) so failures are visible; metrics.json per model.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/exp_001
mkdir -p "$OUT"
: > "$OUT/status.txt"

run() {
  local name="$1"; shift
  echo "$(date +%H:%M:%S) START $name" >> "$OUT/status.txt"
  $PY evaluation/evaluate_baseline.py --baseline "$name" "$@" --outdir "$OUT/$name" \
      > "$OUT/$name.log" 2>&1
  if [ -f "$OUT/$name/metrics.json" ]; then
    echo "$(date +%H:%M:%S) DONE  $name" >> "$OUT/status.txt"
  else
    echo "$(date +%H:%M:%S) FAIL  $name (see $name.log)" >> "$OUT/status.txt"
  fi
}

run YOLOP
run TwinLiteNet
run TwinLiteNetPlus --preset nano
run TwinLiteNetPlus --preset small
run TwinLiteNetPlus --preset medium
run TwinLiteNetPlus --preset large
run TriLiteNet --preset tiny
run TriLiteNet --preset small
run TriLiteNet --preset base
echo "$(date +%H:%M:%S) ALL_DONE" >> "$OUT/status.txt"
