#!/bin/bash
# Every 180s, append a status line to r1r2_monitor.log while training runs.
OUT=/home/mycode/ai_study/trac/experiments/phase2b
LOG="$OUT/r1r2_monitor.log"
rm -f "$LOG"
while pgrep -f "phase2b_run_r1r2.sh|training/train.py" >/dev/null 2>&1; do
  echo "[$(date "+%F %T")] $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) | $(pgrep -c -f train.py 2>/dev/null || echo 0) train | done_rows=$(( $(wc -l < "$OUT/r1r2_results.csv" 2>/dev/null || echo 0) - 1 ))" >> "$LOG"
  # detect which point is active from the newest log
  for l in "$OUT"/r1r2_r2_z*_eval.log "$OUT"/r1r2_r1_z*_eval.log; do :; done
  newest=$(ls -t "$OUT"/r1r2_r*_z*/*.log "$OUT"/r1r2_r*_z*.log 2>/dev/null | head -1)
  active=$(ls -td "$OUT"/r1r2_r*_z* 2>/dev/null | head -1)
  [ -n "$active" ] && echo "  active_dir: $(basename $active)" >> "$LOG"
  sleep 180
done
echo "[$(date "+%F %T")] MONITOR END (runner finished)" >> "$LOG"
