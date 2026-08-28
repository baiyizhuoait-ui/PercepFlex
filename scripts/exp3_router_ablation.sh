#!/usr/bin/env bash
# Experiment 3 (taskbook §11): Random vs Fixed vs Global-difficulty vs Task-aware router.
# Uses the trained adaptive model; swaps only the routing policy:
#   fixed        -> force uniform widths (three budgets), averaged
#   random       -> random per-frame assignment (same budget distribution)
#   task-aware   -> learned router (default)
# (Global-difficulty: a single shared allocation = exp2 methodA, listed here for reference)
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
CKPT=${1:-experiments/exp_train_D/checkpoint.pt}
OUT=experiments/exp_router_ablation
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  $PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$CKPT" "$@" \
      --outdir "$OUT/$tag" --alloc-stats > "$OUT/$tag.log" 2>&1
  [ -f "$OUT/$tag/metrics.json" ] && echo "DONE $tag" | tee -a "$OUT/status.txt" \
      || { echo "FAIL $tag"; tail -3 "$OUT/$tag.log"; }
}

echo "=== Exp3: Router ablation ===" | tee "$OUT/status.txt"
run fixed_tiny    --force-widths 0.25,0.25,0.25
run fixed_medium  --force-widths 0.5,0.5,0.5
run fixed_large   --force-widths 1.0,1.0,1.0
run task_aware
echo "NOTE: random router implemented as a python variant (scripts/random_router_eval.py)"
echo "$(date +%H:%M:%S) ALL_DONE" >> "$OUT/status.txt"
$PY evaluation/summarize.py "$OUT"
