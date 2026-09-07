#!/bin/bash
# Phase 4A Level 2, probe C - Gradient-Magnitude Rebalancing (H11) @ 20 epochs.
#
# Question: is the shared Z being monopolised by detection?
#
# Measured at the Phase 4A 20ep checkpoints, ||dL_t/dZ|| is
#   z16: det 0.0308 / da 0.0059 / lane 0.0054   -> det owns 73.2% of the pull
#   z32: det 0.0250 / da 0.0032 / lane 0.0043   -> det owns 76.9%
# with all pairwise cosines ~0. Under the GradNorm account of multi-task
# training, magnitude imbalance alone (no conflict required) is enough for one
# task to dominate the shared parameters. If that is what is happening here, it
# is a single common cause for three separate observations: DA is insensitive to
# Z, lane only weakly so, and widening Z only ever helps detection.
#
# Intervention: lambda_det = 0.2, lambda_da = lambda_lane = 1.0. Architecture,
# z width, schedule and seed are untouched; the ONLY config change is
# train.lambda_det (asserted by scripts/phase4c_rw_make_configs.py).
#
# Baselines are NOT retrained: the R2 20ep runs (r2_z16, r2_z32) in
# experiments/phase4a/phase4A_results.csv are the same model with lambda_det=1.0.
#
# Usage:
#   bash scripts/phase4c_rw_run.sh
#   CELLS_OVERRIDE="rw_z32" EPOCHS_OVERRIDE=4 bash scripts/phase4c_rw_run.sh
#
# Control:  touch experiments/phase4a/STOP_C to halt before the next cell.
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

SEED=${SEED_OVERRIDE:-0}
EP=${EPOCHS_OVERRIDE:-20}
CELLS=(${CELLS_OVERRIDE:-rw_z16 rw_z32})

OUT=experiments/phase4a
mkdir -p "$OUT"
CSV=${CSV_OVERRIDE:-$OUT/phase4C_probeC_results.csv}
RUNNER_LOG="$OUT/exp4C_runner.log"
STOP_FILE="$OUT/STOP_C"
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

HEADER="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

echo "===== [$(date '+%F %T')] PHASE 4C PROBE C (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
echo "cells=${CELLS[*]}  epochs=${EP}  seed=${SEED}  csv=${CSV}" | tee -a "$RUNNER_LOG"

extract_train_stats() {
  local LOG="$1" PEAK_MEM="NA" FINAL_LOSS="NA"
  if [ -f "$LOG" ]; then
    PEAK_MEM=$(grep -oE 'mem [0-9]+/[0-9]+MiB' "$LOG" | awk '{print $2}' | cut -d/ -f1 | sort -n | tail -1)
    if [ -z "$PEAK_MEM" ]; then
      PEAK_MEM=$(awk '{for(i=1;i<=NF;i++) if($i=="mem") print $(i+1)}' "$LOG" | cut -d/ -f1 | sort -n | tail -1)
    fi
    [ -z "$PEAK_MEM" ] && PEAK_MEM="NA"
    FINAL_LOSS=$(grep -oE 'avg_loss=[0-9.]+' "$LOG" | tail -1 | sed 's/avg_loss=//')
    [ -z "$FINAL_LOSS" ] && FINAL_LOSS="NA"
  fi
  echo "${PEAK_MEM} ${FINAL_LOSS}"
}

PROBE="$OUT/.ckpt_probe_C.py"
cat > "$PROBE" <<'PYEOF'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(ck.get("epoch", "ERR"))
except Exception:
    print("ERR")
PYEOF

for CELL in "${CELLS[@]}"; do
  if [ -f "$STOP_FILE" ]; then
    echo "===== [$(date '+%F %T')] STOP_C present - halt before ${CELL} =====" | tee -a "$RUNNER_LOG"
    exit 0
  fi

  Z="${CELL##*_z}"
  CFG="configs/phase4c_${CELL}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[WARN] missing config $CFG - skip ${CELL}" | tee -a "$RUNNER_LOG"
    continue
  fi

  TAG="exp4C_${CELL}_e${EP}"
  TR="$OUT/$TAG"; EVL="$OUT/${TAG}_eval"
  CKPT="$TR/checkpoint.pt"; MP="$EVL/metrics.json"; TLOG="$TR/training_log.txt"
  mkdir -p "$TR" "$EVL"

  echo "===== [$(date '+%F %T')] ---- ${CELL} (${EP}ep) ---- =====" | tee -a "$RUNNER_LOG"

  CK_EP=""
  [ -f "$CKPT" ] && CK_EP=$("$PY" "$PROBE" "$CKPT" 2>/dev/null)

  WALL_MIN="NA"
  if [ -f "$MP" ]; then
    echo "[resume] ${TAG} metrics present - append without retraining" | tee -a "$RUNNER_LOG"
  elif [ -f "$CKPT" ] && [ "$CK_EP" = "$EP" ]; then
    echo "[resume] ${TAG} ckpt epoch=${CK_EP} matches target - eval only" | tee -a "$RUNNER_LOG"
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" \
      2>&1 | tee -a "$EVL.log" > "$EVL/eval_stdout.log"
  else
    if [ -f "$CKPT" ]; then
      echo "[resume] ${TAG} ckpt epoch=${CK_EP} != ${EP} - DISCARD partial, retrain" | tee -a "$RUNNER_LOG"
      mv "$CKPT" "${CKPT}.partial_ep${CK_EP}"
    fi
    T0=$(date +%s)
    echo "===== [$(date '+%F %T')] ${CELL} TRAIN start (epochs=${EP} seed=${SEED}) =====" | tee -a "$RUNNER_LOG"
    "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs "$EP" --seed "$SEED" \
      2>&1 | tee -a "$TLOG" > "$TR/train_stdout.log"
    T1=$(date +%s)
    WALL_MIN=$(( (T1 - T0) / 60 ))
    echo "===== [$(date '+%F %T')] ${CELL} TRAIN done in ${WALL_MIN} min =====" | tee -a "$RUNNER_LOG"

    if [ -f "$CKPT" ]; then
      echo "===== [$(date '+%F %T')] ${CELL} EVAL =====" | tee -a "$RUNNER_LOG"
      "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" \
        2>&1 | tee -a "$EVL.log" > "$EVL/eval_stdout.log"
    fi
  fi

  if [ ! -f "$MP" ]; then
    echo "[FAIL] ${TAG} produced no metrics.json" | tee -a "$RUNNER_LOG"
    echo "r2rw,${CELL},${Z},ebase,${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},failed-no-metrics,${COMMIT}" >> "$CSV"
    continue
  fi

  CK_EP=$("$PY" "$PROBE" "$CKPT" 2>/dev/null)
  if [ "$CK_EP" != "$EP" ]; then
    echo "[FAIL] ${TAG} checkpoint epoch=${CK_EP} != ${EP} - row flagged" | tee -a "$RUNNER_LOG"
  else
    echo "[ok] ${TAG} checkpoint epoch=${CK_EP} == ${EP}" | tee -a "$RUNNER_LOG"
  fi

  read -r PEAK_MEM FINAL_LOSS <<< "$(extract_train_stats "$TLOG")"

  "$PY" - "$CELL" "$Z" "$EP" "$SEED" "$COMMIT" "$CSV" "$MP" \
        "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
(cell, z, ep, seed, commit, csv, mp, wall, mem, floss) = sys.argv[1:11]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if f != f:
        return "NA"
    return "%.4f" % f
row = ["r2rw", cell, z, "ebase", ep,
       num(g("parameters") / 1e6 if g("parameters") else "NA"),
       num(g("flops") / 1e9 if g("flops") else "NA"),
       "%.2f" % g("fps"),
       num(g("mAP50")), num(g("mAP50_95")),
       num(g("da_mIoU")), num(g("da_fg_iou")),
       num(g("lane_mIoU")), num(g("lane_fg_iou")),
       str(mem), str(floss), str(wall), seed, "trained", commit]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF

  echo "[ok] ${TAG} params=$("$PY" -c "import json;print(round(json.load(open('$MP'))['parameters']/1e6,4))")M " \
       "flops=$("$PY" -c "import json;print(round(json.load(open('$MP'))['flops']/1e9,3))")G " \
       "mAP50=$("$PY" -c "import json;print(json.load(open('$MP'))['mAP50'])") " \
       "da=$("$PY" -c "import json;print(json.load(open('$MP'))['da_mIoU'])") " \
       "lane=$("$PY" -c "import json;print(json.load(open('$MP'))['lane_mIoU'])")" | tee -a "$RUNNER_LOG"
done

rm -f "$PROBE"
echo "===== [$(date '+%F %T')] PROBE C RUNNER FINISHED =====" | tee -a "$RUNNER_LOG"
