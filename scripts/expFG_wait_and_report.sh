#!/usr/bin/env bash
# Wait for the F/G pipeline to finish, then produce the analysis.
set -u
cd "$(dirname "$0")/.."
OUT=experiments/expF_single_teacher
# wait for all three teacher trainings + evals (status shows EXP-F/G DONE)
for i in $(seq 1 180); do
  if grep -q "EXP-F/G DONE" "$OUT/status.txt" 2>/dev/null; then
    echo "$(date +%H:%M:%S) F/G pipeline done"
    break
  fi
  sleep 120
done
echo "=== F/G analysis ==="
../gpu_env/bin/python scripts/expFG_analyze.py 2>&1 | tee "$OUT/analysis.txt"
echo "$(date +%H:%M:%S) F/G ANALYZED"
