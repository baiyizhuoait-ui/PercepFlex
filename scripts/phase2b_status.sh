#!/bin/bash
# Phase-2-B — one-shot status board.
#
# Shows: GPU state, which R0/R1/R2 points are complete vs pending, and the tail
# of whatever is currently running. Read-only: never starts or stops anything.
#
#   bash scripts/phase2b_status.sh
set -u
cd /home/mycode/ai_study/trac || exit 1
OUT=experiments/phase2b

echo "=================== GPU ==================="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,clocks_throttle_reasons.active \
  --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"

echo
echo "=================== RUNNING ==================="
pgrep -af "training/train.py|evaluate_baseline.py|phase2b_run_" 2>/dev/null \
  || echo "(nothing running)"

echo
echo "=================== STOP FILE ==================="
if [ -f "$OUT/STOP" ]; then echo "PRESENT -> runners will halt at the next point boundary"; \
else echo "absent -> runners may start the next point"; fi

echo
echo "=========== R0 Z-SWEEP (zsweep_results.csv) ==========="
if [ -f "$OUT/zsweep_results.csv" ]; then
  column -s, -t "$OUT/zsweep_results.csv" 2>/dev/null | cut -c1-140 \
    || cat "$OUT/zsweep_results.csv"
else echo "(missing)"; fi

echo
echo "=========== R1/R2 (r1r2_results.csv) ==========="
if [ -f "$OUT/r1r2_results.csv" ]; then
  column -s, -t "$OUT/r1r2_results.csv" 2>/dev/null | cut -c1-160 \
    || cat "$OUT/r1r2_results.csv"
else echo "(missing)"; fi

echo
echo "=========== CHECKPOINTS ON DISK ==========="
for d in "$OUT"/r1r2_*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  ck="no"; mp="no"
  [ -f "$d/checkpoint.pt" ] && ck="yes"
  [ -f "$d/metrics.json" ] && mp="yes"
  printf "  %-22s checkpoint=%-3s metrics=%s\n" "$n" "$ck" "$mp"
done
[ -d "$OUT/r1r2_r2_z32" ] || echo "  (no r1r2_* dirs yet)"

echo
echo "=========== LAST LOG LINES ==========="
for f in "$OUT"/r1r2_*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f")"
  tail -n 3 "$f"
done
