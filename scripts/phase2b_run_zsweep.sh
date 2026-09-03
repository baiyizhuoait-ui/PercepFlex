#!/bin/bash
# Phase-2-B Task 2(a) Z-capacity sweep runner.
# Trains + evaluates Z in {16,32,48,64,96,128} with a FIXED encoder (0.235M geometry),
# fixed Stage-A protocol (4ep, bs16, AdamW lr1e-3, cosine, tri_train full, seed0).
# Each run: train -> eval(OursStatic) -> append one row to zsweep_results.csv.
#
# Launch in background on the WSL box (single RTX5060, do NOT run two sweeps in parallel):
#   nohup bash scripts/phase2b_run_zsweep.sh > experiments/phase2b/zsweep_runner.log 2>&1 &
#
# Expected wall time: ~6 x (train ~28-36min + eval ~8min) ~= 4h. Poll with:
#   tail -n 20 experiments/phase2b/zsweep_runner.log
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

ZS=(16 32 48 64 96 128)
OUT=experiments/phase2b
mkdir -p "$OUT"
CSV="$OUT/zsweep_results.csv"
RUNNER_LOG="$OUT/zsweep_runner.log"

HEADER="z,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_fg,lane_mIoU,checkpoint,seed,epochs,git_commit"
: > "$CSV"
echo "$HEADER" > "$CSV"

for Z in "${ZS[@]}"; do
  CFG="configs/phase2b_zsweep_z${Z}.yaml"
  TR="$OUT/zsweep_z${Z}"
  EVL="$OUT/zsweep_z${Z}_eval"
  CKPT="$TR/checkpoint.pt"
  COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

  echo "===== [$(date '+%F %T')] Z=${Z} TRAIN (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
  "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs 4 --seed 0 2>&1 | tee -a "$TR.log"

  if [ ! -f "$CKPT" ]; then
    echo "[WARN] Z=${Z} training did not produce $CKPT — skipping eval, recording NaN row." | tee -a "$RUNNER_LOG"
    echo "${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},0,4,${COMMIT}" >> "$CSV"
    continue
  fi

  echo "===== [$(date '+%F %T')] Z=${Z} EVAL =====" | tee -a "$RUNNER_LOG"
  "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" 2>&1 | tee -a "$EVL.log"

  MP="$EVL/metrics.json"
  if [ ! -f "$MP" ]; then
    echo "[WARN] Z=${Z} eval metrics missing — recording NaN row." | tee -a "$RUNNER_LOG"
    echo "${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},0,4,${COMMIT}" >> "$CSV"
    continue
  fi

  "$PY" - "$Z" "$CKPT" "$COMMIT" "$CSV" "$MP" <<'PYEOF'
import json, sys
z, ckpt, commit, csv, mp = sys.argv[1:6]
m = json.load(open(mp))
def g(k, d=0.0):
    return m.get(k, d)
row = [
    z,
    f"{g('parameters')/1e6:.3f}",
    f"{g('flops')/1e9:.3f}",
    f"{g('fps'):.2f}",
    f"{g('mAP50'):.4f}",
    f"{g('mAP50_95'):.4f}",
    f"{g('da_mIoU'):.4f}",
    f"{g('da_fg_iou'):.4f}",
    f"{g('lane_fg_iou'):.4f}",
    f"{g('lane_mIoU'):.4f}",
    ckpt, "0", "4", commit,
]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF
done

echo "===== [$(date '+%F %T')] Z-SWEEP DONE =====" | tee -a "$RUNNER_LOG"
echo "Results CSV: $CSV" | tee -a "$RUNNER_LOG"
"$PY" scripts/phase2b_analyze.py --root "$ROOT" --mode auto \
  --out "$OUT/zsweep_analysis.md" 2>&1 | tee -a "$RUNNER_LOG"
