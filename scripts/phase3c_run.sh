#!/bin/bash
# Phase 3C · Fixed-Budget Capacity Allocation -- budget-matching runner.
#
# Trains ONLY the two new cells needed to complete a strict equal-parameter
# triad inside Budget-L and Budget-M:
#
#   Budget-L (~0.19M)  esmall_z128 [Z-heavy]   ebase_z16  [encoder-heavy]  midL_z32  [balanced] <- NEW
#   Budget-M (~0.29M)  ebase_z128  [Z-heavy]   elarge_z16 [encoder-heavy]  midM_z32  [balanced] <- NEW
#
# The two new encoders are pure width interpolations of the existing scaling
# law (stem/stages only). blocks [2,2,2], depth, heads, losses untouched.
# See scripts/phase3c_design_encoders.py and phase3C_config_audit.txt.
#
# Frozen protocol: seed 0, epochs 20, batch 16, lr 1e-3, AdamW + cosine,
# tri_train 69863 imgs, 640x640. Only encoder width and Z width move.
#
# Execution order is ascending compute cost. That is an execution detail only.
#
# Output: experiments/phase3c/phase3C_new_cells.csv
#         experiments/phase3c/exp3C_<cell>/          (ckpt + logs)
#         experiments/phase3c/exp3C_<cell>_eval/     (metrics.json)
#
# Control:  touch experiments/phase3c/STOP to halt before the next cell.
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

# The config audit greps these two lines to prove the protocol is pinned.
SEED=${SEED_OVERRIDE:-0}
EP=${EPOCHS_OVERRIDE:-20}

CELLS=(${CELLS_OVERRIDE:-midL_z32 midM_z32})

OUT=experiments/phase3c
mkdir -p "$OUT"
CSV="$OUT/phase3C_new_cells.csv"
RUNNER_LOG="$OUT/exp3C_runner.log"
STOP_FILE="$OUT/STOP"
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

HEADER="encoder,z,stem,stages,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

echo "===== [$(date '+%F %T')] PHASE 3C BUDGET-MATCH RUNNER (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
echo "cells=${CELLS[*]}  epochs=${EP}  seed=${SEED}" | tee -a "$RUNNER_LOG"

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

  ENC="${CELL%_*}"
  Z="${CELL##*_z}"

  CFG="configs/phase3c_${ENC}_z${Z}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[WARN] missing config $CFG - skip ${CELL}" | tee -a "$RUNNER_LOG"
    continue
  fi

  TAG="exp3C_${CELL}"
  TR="$OUT/$TAG"
  EVL="$OUT/${TAG}_eval"
  CKPT="$TR/checkpoint.pt"
  MP="$EVL/metrics.json"
  TLOG="$TR/training_log.txt"
  mkdir -p "$TR" "$EVL"

  echo "===== [$(date '+%F %T')] ---- ${CELL} ---- =====" | tee -a "$RUNNER_LOG"

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
    echo "${ENC},${Z},NA,NA,${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},failed-no-metrics,${COMMIT}" >> "$CSV"
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
    WALL_MIN=$(grep "TRAIN done in" "$RUNNER_LOG" | grep "${CELL} " | tail -1 | grep -oE 'in [0-9]+ min' | grep -oE '[0-9]+')
    [ -z "$WALL_MIN" ] && WALL_MIN="NA"
  fi

  STEM=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['encoder']['stem'])" "$CFG")
  STAGES=$("$PY" -c "import yaml,sys;print('-'.join(str(c) for c in yaml.safe_load(open(sys.argv[1]))['model']['encoder']['stages']))" "$CFG")

  "$PY" - "$ENC" "$Z" "$STEM" "$STAGES" "$EP" "$SEED" "$CKPT" "$COMMIT" "$CSV" "$MP" \
        "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
(enc, z, stem, stages, ep, seed, ckpt, commit, csv, mp,
 wall, mem, floss) = sys.argv[1:14]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
row = [enc, z, stem, stages, ep,
       f"{g('parameters')/1e6:.4f}", f"{g('flops')/1e9:.4f}", f"{g('fps'):.2f}",
       f"{g('mAP50'):.4f}", f"{g('mAP50_95'):.4f}",
       f"{g('da_mIoU'):.4f}", f"{g('da_fg_iou'):.4f}",
       f"{g('lane_mIoU'):.4f}", f"{g('lane_fg_iou'):.4f}",
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
echo "===== [$(date '+%F %T')] PHASE 3C BUDGET-MATCH RUNNER DONE =====" | tee -a "$RUNNER_LOG"
