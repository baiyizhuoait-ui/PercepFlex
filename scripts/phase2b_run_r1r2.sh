#!/bin/bash
# Phase-2-B Task 2(c): R1/R2 — does Detection belong in Compact Z?
#   R1 (z_proj=false): detection reads Z directly.
#   R2 (z_proj=true):  detection reads Z through a 1x1 projection (PRIORITY).
# Both are compared against R0 (results in zsweep_results.csv), which reads the
# encoder's multi-scale features instead of Z.
#
# Same fixed protocol as the R0 sweep: 4ep, bs16, AdamW lr1e-3, cosine, tri_train.
#
# Usage (single GPU — never run concurrently with the R0 sweep):
#   bash scripts/phase2b_run_r1r2.sh              # default: R2 @ 32,64,96 + R1 @ 64
#   bash scripts/phase2b_run_r1r2.sh "R2:32,64,96"
#   bash scripts/phase2b_run_r1r2.sh "R1:64" "R2:32,64,96"
#
# Control (mirrors scripts/phase2b_run_zsweep_v2.sh):
#   1. GRACEFUL STOP: drop a file at experiments/phase2b/STOP to halt cleanly when the
#      current unit finishes. The running train/eval is never killed mid-way.
#          touch experiments/phase2b/STOP      # stops before the next unit
#          rm   experiments/phase2b/STOP       # re-enable
#   2. RESUME: a unit is skipped when r1r2_<tag>_eval/metrics.json already exists.
#      If only checkpoint.pt exists (train done, eval missing), training is skipped and
#      the run goes straight to eval. Re-running the same command is therefore safe.
#   3. The results CSV is APPENDED to, never truncated, so a restart keeps prior rows.
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

SPECS=("$@")
if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=("R2:32,64,96" "R1:64")
fi

OUT=experiments/phase2b
mkdir -p "$OUT"
CSV="$OUT/r1r2_results.csv"
RUNNER_LOG="$OUT/r1r2_runner.log"
HEADER="variant,z,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_fg,lane_mIoU,checkpoint,seed,epochs,git_commit"
# APPEND, never truncate: restarting must not destroy already-collected rows.
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"
STOP_FILE="$OUT/STOP"

for SPEC in "${SPECS[@]}"; do
  V="${SPEC%%:*}"            # R1 or R2
  ZLIST="${SPEC#*:}"
  VL=$(echo "$V" | tr 'A-Z' 'a-z')
  IFS=',' read -ra ZS <<< "$ZLIST"
  for Z in "${ZS[@]}"; do
    if [ -f "$STOP_FILE" ]; then
      echo "===== [$(date '+%F %T')] STOP FILE PRESENT — halting before ${V} Z=${Z} =====" | tee -a "$RUNNER_LOG"
      echo "Remove $STOP_FILE to continue." | tee -a "$RUNNER_LOG"
      break 2
    fi
    CFG="configs/phase2b_${VL}_zsweep_z${Z}.yaml"
    if [ ! -f "$CFG" ]; then
      echo "[WARN] missing config $CFG — skipping" | tee -a "$RUNNER_LOG"
      continue
    fi
    TAG="${VL}_z${Z}"
    TR="$OUT/r1r2_${TAG}"
    EVL="$OUT/r1r2_${TAG}_eval"
    CKPT="$TR/checkpoint.pt"
    MP="$EVL/metrics.json"
    COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    # RESUME: fully evaluated -> skip the unit entirely.
    if [ -f "$MP" ]; then
      echo "[resume] ${TAG} already evaluated — skipping" | tee -a "$RUNNER_LOG"
      continue
    fi

    # RESUME: trained but never evaluated -> eval only (saves a full training run).
    if [ -f "$CKPT" ]; then
      echo "[resume] ${TAG} checkpoint exists — skipping training, going straight to eval" | tee -a "$RUNNER_LOG"
    else
      echo "===== [$(date '+%F %T')] ${V} Z=${Z} TRAIN (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
      "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs 4 --seed 0 2>&1 | tee -a "$TR.log"
    fi

    if [ ! -f "$CKPT" ]; then
      echo "[WARN] ${TAG} train produced no checkpoint — NaN row" | tee -a "$RUNNER_LOG"
      echo "${V},${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},0,4,${COMMIT}" >> "$CSV"
      continue
    fi

    echo "===== [$(date '+%F %T')] ${V} Z=${Z} EVAL =====" | tee -a "$RUNNER_LOG"
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" \
      --outdir "$EVL" 2>&1 | tee -a "$EVL.log"

    if [ ! -f "$MP" ]; then
      echo "[WARN] ${TAG} eval metrics missing — NaN row" | tee -a "$RUNNER_LOG"
      echo "${V},${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},0,4,${COMMIT}" >> "$CSV"
      continue
    fi

    "$PY" - "$V" "$Z" "$CKPT" "$COMMIT" "$CSV" "$MP" <<'PYEOF'
import json, sys
v, z, ckpt, commit, csv, mp = sys.argv[1:7]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
row = [v, z,
       f"{g('parameters')/1e6:.3f}", f"{g('flops')/1e9:.3f}", f"{g('fps'):.2f}",
       f"{g('mAP50'):.4f}", f"{g('mAP50_95'):.4f}",
       f"{g('da_mIoU'):.4f}", f"{g('da_fg_iou'):.4f}",
       f"{g('lane_fg_iou'):.4f}", f"{g('lane_mIoU'):.4f}",
       ckpt, "0", "4", commit]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF
  done
done

if [ -f "$STOP_FILE" ]; then
  echo "===== [$(date '+%F %T')] HALTED BY STOP FILE =====" | tee -a "$RUNNER_LOG"
  echo "Remove $STOP_FILE to continue." | tee -a "$RUNNER_LOG"
else
  echo "===== [$(date '+%F %T')] R1/R2 DONE =====" | tee -a "$RUNNER_LOG"
fi
echo "Results CSV: $CSV" | tee -a "$RUNNER_LOG"
