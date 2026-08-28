#!/usr/bin/env bash
# Experiment 2 (taskbook §11): Shared Dynamic vs Task-wise Dynamic.
# Method A: one shared allocation for all tasks (router.shared=True).
# Method B: per-task independent allocation (default router).
# Compares two trained adaptive checkpoints on tri_val.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
CKPT_A=${1:-experiments/exp_train_D_shared/checkpoint.pt}
CKPT_B=${2:-experiments/exp_train_D/checkpoint.pt}
OUT=experiments/exp_shared_vs_taskwise
mkdir -p "$OUT"

run() {
  local tag="$1" ckpt="$2"; shift 2
  $PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$ckpt" \
      --outdir "$OUT/$tag" --alloc-stats > "$OUT/$tag.log" 2>&1
  [ -f "$OUT/$tag/metrics.json" ] && echo "DONE $tag" | tee -a "$OUT/status.txt" \
      || { echo "FAIL $tag"; tail -3 "$OUT/$tag.log"; }
}

echo "=== Exp2: Shared vs Task-wise ===" | tee "$OUT/status.txt"
run methodA_shared "$CKPT_A"
run methodB_taskwise "$CKPT_B"
echo "=== summary ==="
$PY evaluation/summarize.py "$OUT"
echo "=== allocation stats (task heterogeneity) ==="
$PY - "$OUT" <<'PYEOF'
import csv, os, sys
base = sys.argv[1]
for tag in ("methodA_shared", "methodB_taskwise"):
    p = os.path.join(base, tag, "allocation_statistics.csv")
    if not os.path.exists(p):
        print(tag, "no allocation csv"); continue
    rows = list(csv.DictReader(open(p)))
    het = sum(1 for r in rows if len({r["det"], r["da"], r["lane"]}) > 1) / len(rows)
    print(f"{tag}: frames={len(rows)} task-heterogeneity={het:.3f}")
PYEOF
