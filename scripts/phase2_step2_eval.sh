#!/usr/bin/env bash
# Phase-2 Step 2: evaluate the paired no-KD vs single-teacher KD models on full tri_val.
set -u
cd "$(dirname "$0")/.."
PY=/home/mycode/ai_study/gpu_env/bin/python
S2=experiments/phase2/s2
EV=experiments/phase2/s2_eval
mkdir -p "$EV"
st(){ echo "$(date +%H:%M:%S) $*" >> "$EV/status.txt"; }

for m in s2_nokd s2_kd_YOLOP s2_kd_TLP s2_kd_TriLite; do
  if [ -f "$EV/$m/metrics.json" ]; then st "$m eval exists"; continue; fi
  st "START $m"
  $PY evaluation/evaluate_baseline.py --baseline OursStatic --preset "$S2/$m/checkpoint.pt" \
      --outdir "$EV/$m" > "$EV/$m.log" 2>&1
  [ -f "$EV/$m/metrics.json" ] && st "DONE $m" || st "FAIL $m"
done
st "STEP2_EVAL_DONE"
