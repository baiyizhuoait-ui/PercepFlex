#!/usr/bin/env bash
# Rerun the presets that were lost to the runner bug (TLP small/med/large, TriLite small/base).
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
OUT=experiments/exp_001

run() {
  local name="$1"; local preset="$2"; shift 2
  local tag="${name}-${preset}"
  echo "$(date +%H:%M:%S) START $tag" >> "$OUT/status.txt"
  $PY evaluation/evaluate_baseline.py --baseline "$name" --preset "$preset" --outdir "$OUT/$tag" \
      > "$OUT/$tag.log" 2>&1
  [ -f "$OUT/$tag/metrics.json" ] && echo "$(date +%H:%M:%S) DONE  $tag" >> "$OUT/status.txt" \
      || echo "$(date +%H:%M:%S) FAIL  $tag" >> "$OUT/status.txt"
}

run TwinLiteNetPlus small
run TwinLiteNetPlus medium
run TwinLiteNetPlus large
run TriLiteNet small
run TriLiteNet base
echo "$(date +%H:%M:%S) ALL_DONE" >> "$OUT/status.txt"
