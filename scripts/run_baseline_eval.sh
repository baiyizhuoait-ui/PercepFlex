#!/usr/bin/env bash
# Phase 1 Step 3: full unified baseline evaluation on tri_val (10,000 images).
# Runs all baselines (official weights) and writes metrics.json per model into
# experiments/exp_001/. Logs to experiments/exp_001/training_log.txt.
set -e
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/exp_001
mkdir -p "$OUT"

run() {
  local name="$1"; shift
  echo "=== $(date +%H:%M:%S) $name ===" | tee -a "$OUT/training_log.txt"
  $PY evaluation/evaluate_baseline.py --baseline "$name" "$@" \
      --outdir "$OUT/$name" 2>&1 | grep -vE "UserWarning|warnings.warn" \
      | tee -a "$OUT/training_log.txt"
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
echo "=== ALL DONE $(date +%H:%M:%S) ===" | tee -a "$OUT/training_log.txt"
