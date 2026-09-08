#!/bin/bash
# Phase 5 probe completion watcher.
# Polls every 60s for all three probe cells in the CSV. When present, runs
# the post-probe pipeline and exits. Logs to experiments/phase5/exp5_watcher.log.

set -u
PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
CSV=$ROOT/experiments/phase5/phase5_lane_probe_results.csv
DONE_FLAG=$ROOT/experiments/phase5/exp5_postprobe_done.flag
LOG=$ROOT/experiments/phase5/exp5_watcher.log

exec >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] watcher started, polling $CSV"

while true; do
  if [ -f "$DONE_FLAG" ]; then
    echo "[$(date +%H:%M:%S)] $DONE_FLAG exists, exiting"
    exit 0
  fi
  n=$(grep -E "^(r2u_z16|l14up_z16|l14f1_z16|lch64_z16)," "$CSV" 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then
    echo "[$(date +%H:%M:%S)] all four cells present ($n), running orchestrator"
    cd "$ROOT" && "$PY" scripts/phase5_postprobe.py
    touch "$DONE_FLAG"
    echo "[$(date +%H:%M:%S)] orchestrator done, flag set, exiting"
    exit 0
  fi
  sleep 60
done
