#!/bin/bash
# Phase 4A · Shared-Bottleneck (R2) runner.
#
# R2 forces detection through the compact representation Z, removing the R0
# bypass that was confirmed by intervention in
# experiments/phase4a/shared_bottleneck_trace.txt.
#
# Encoder is PINNED to E-base (stem 16 / stages [32,64,96,128] / blocks [2,2,2])
# because it is the only encoder with a complete 20ep R0 triple at z16/z32/z128,
# which is exactly what a paired R0-vs-R2 comparison needs.
#
# The only difference from the R0 runs that already exist is the detection
# information path (model.detection.from_z / z_proj / det_ch), proven by
# scripts/phase4a_make_configs.py. z_channels, encoder, heads, losses, data,
# augmentation, seed, batch, lr, input size are all unchanged.
#
# Usage:
#   EPOCHS_OVERRIDE=4  CELLS_OVERRIDE="r2_z16" bash scripts/phase4a_run.sh   # sanity
#   CELLS_OVERRIDE="r2_z16 r2_z32 r2_z128" bash scripts/phase4a_run.sh       # main (20ep)
#
# Control:  touch experiments/phase4a/STOP to halt before the next cell.
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

# The config audit greps these two lines to prove the protocol is pinned.
SEED=${SEED_OVERRIDE:-0}
EP=${EPOCHS_OVERRIDE:-20}

CELLS=(${CELLS_OVERRIDE:-r2_z16 r2_z32 r2_z128})

OUT=experiments/phase4a
mkdir -p "$OUT"
if [ "$EP" = "20" ]; then
  CSV="$OUT/phase4A_results.csv"
else
  CSV="$OUT/phase4A_sanity.csv"
fi
CSV=${CSV_OVERRIDE:-$CSV}
RUNNER_LOG="$OUT/exp4A_runner.log"
STOP_FILE="$OUT/STOP"
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

HEADER="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

echo "===== [$(date '+%F %T')] PHASE 4A RUNNER (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
echo "cells=${CELLS[*]}  epochs=${EP}  seed=${SEED}  csv=${CSV}" | tee -a "$RUNNER_LOG"

# peak GPU mem + final train loss.
# WARNING: keep `cut -d/ -f1`. Log lines read "mem 2700/8151MiB"; dropping the
# slash first glues numerator to denominator -> 27008151, poisoning the table.
extract_train_stats() {
  local LOG="$1"
  local PEAK_MEM="NA"
  local FINAL_LOSS="NA"
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

PROBE="$OUT/.ckpt_probe.py"
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
    echo "===== [$(date '+%F %T')] STOP present - halt before ${CELL} =====" | tee -a "$RUNNER_LOG"
    exit 0
  fi

  VAR="${CELL%%_*}"          # r2
  Z="${CELL##*_z}"           # 16 / 32 / 128

  CFG="configs/phase4a_${VAR}_z${Z}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[WARN] missing config $CFG - skip ${CELL}" | tee -a "$RUNNER_LOG"
    continue
  fi

  TAG="exp4A_${CELL}_e${EP}"
  TR="$OUT/$TAG"
  EVL="$OUT/${TAG}_eval"
  CKPT="$TR/checkpoint.pt"
  MP="$EVL/metrics.json"
  TLOG="$TR/training_log.txt"
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
    echo "${VAR},${CELL},${Z},ebase,${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},failed-no-metrics,${COMMIT}" >> "$CSV"
    continue
  fi

  CK_EP=$("$PY" "$PROBE" "$CKPT" 2>/dev/null)
  if [ "$CK_EP" != "$EP" ]; then
    echo "[FAIL] ${TAG} checkpoint epoch=${CK_EP} != ${EP} - row flagged" | tee -a "$RUNNER_LOG"
  else
    echo "[ok] ${TAG} checkpoint epoch=${CK_EP} == ${EP}" | tee -a "$RUNNER_LOG"
  fi

  read -r PEAK_MEM FINAL_LOSS <<< "$(extract_train_stats "$TLOG")"
  if [ "$WALL_MIN" = "NA" ]; then
    WALL_MIN="NA"
  fi

  "$PY" - "$VAR" "$CELL" "$Z" "$EP" "$SEED" "$COMMIT" "$CSV" "$MP" \
        "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
(var, cell, z, ep, seed, commit, csv, mp, wall, mem, floss) = sys.argv[1:12]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
# NaN guard: a collapsed run must not silently enter the table as a number.
def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    if f != f:
        return "NA"
    return "%.4f" % f
row = [var, cell, z, "ebase", ep,
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
if [ -f "$STOP_FILE" ]; then
  echo "===== [$(date '+%F %T')] HALTED BY STOP =====" | tee -a "$RUNNER_LOG"
  exit 0
fi
echo "===== [$(date '+%F %T')] PHASE 4A RUNNER DONE =====" | tee -a "$RUNNER_LOG"
