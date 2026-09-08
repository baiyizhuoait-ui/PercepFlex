#!/bin/bash
# Phase 2-C Exp A monitor: every 240s append a status line while the runner lives.
OUT=/home/mycode/ai_study/trac/experiments/phase2c
LOG="$OUT/expA_monitor.log"
rm -f "$LOG"
while pgrep -f "phase2c_run_expA.sh|training/train.py" >/dev/null 2>&1; do
  rows=$(( $(wc -l < "$OUT/expA_multiseed.csv" 2>/dev/null || echo 0) - 1 ))
  done_dirs=$(ls -d "$OUT"/expA_z*_eval 2>/dev/null | wc -l)
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
  active=$(ls -td "$OUT"/expA_z*_s* 2>/dev/null | head -1)
  act=$(basename "$active" 2>/dev/null || echo "-")
  echo "[$(date '+%F %T')] gpu=$gpu | csv_rows=$rows | done=$done_dirs/6 | active=$act" >> "$LOG"
  sleep 240
done
echo "[$(date '+%F %T')] EXP-A MONITOR END (runner finished)" >> "$LOG"
