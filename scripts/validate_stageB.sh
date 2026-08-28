#!/usr/bin/env bash
# Stage B quick validation: init adaptive from Stage A ckpt, train 1 epoch on a
# subset, then eval on val + report router allocation statistics.
set -u
cd "$(dirname "$0")/.."
PY=../gpu_env/bin/python
A_CKPT=${1:-experiments/exp_train_A/checkpoint.pt}
N=${2:-8000}
OUT=experiments/exp_train_B_val
$PY training/train.py --config configs/train_stageB.yaml --num-images "$N" --epochs 1 \
    --init "$A_CKPT" --outdir "$OUT"
echo "=== eval (dynamic, 1000 val imgs, alloc stats) ==="
$PY evaluation/evaluate_baseline.py --baseline OursDynamic --preset "$OUT/checkpoint.pt" \
    --num-images 1000 --outdir "$OUT/eval_dynamic" --alloc-stats
echo "=== allocation statistics ==="
python3 - "$OUT/eval_dynamic/allocation_statistics.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
det = [r["det"] for r in rows]; da = [r["da"] for r in rows]; lane = [r["lane"] for r in rows]
from collections import Counter
print("det :", dict(sorted(Counter(det).items())))
print("da  :", dict(sorted(Counter(da).items())))
print("lane:", dict(sorted(Counter(lane).items())))
het = sum(1 for r in rows if len({r["det"], r["da"], r["lane"]}) > 1) / len(rows)
print(f"task-heterogeneity: {het:.3f}  ({len(rows)} frames)")
PYEOF
