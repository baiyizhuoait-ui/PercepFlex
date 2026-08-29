#!/usr/bin/env bash
# Phase 1-B master chain: wait for F/G -> launch I -> final report.
set -u
cd "$(dirname "$0")/.."
OUT=experiments/expF_single_teacher
for i in $(seq 1 240); do
  if grep -q "EXP-F/G DONE" "$OUT/status.txt" 2>/dev/null; then
    echo "$(date +%H:%M:%S) F/G done, launching Exp I"
    break
  fi
  sleep 120
done
echo "$(date +%H:%M:%S) F/G analysis:"
../gpu_env/bin/python scripts/expFG_analyze.py 2>&1 | tee "$OUT/analysis.txt"
echo "$(date +%H:%M:%S) launching Exp I (KD-dynamic)"
./scripts/expI_kd_dynamic.sh
echo "$(date +%H:%M:%S) PHASE1B CHAIN COMPLETE"
