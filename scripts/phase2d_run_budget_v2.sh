#!/bin/bash
# Phase 2-D · Budget-sensitivity experiment.
#
# Purpose: check whether the 4ep Z-sweep signal was masked by training budget.
# Design: fixed seed=0, batch=16, lr=1e-3, encoder fixed (stem16/stages[32,64,96,128]).
# Cells : z ∈ {16, 32, 128} × epochs ∈ {4, 10, 20}
#   - 4ep cells are the existing Phase 2-B zsweep (seed0) runs; re-eval is NOT needed.
#   - 10ep and 20ep cells are new training.
#
# Output : experiments/phase2d/expD_budget.csv
#   columns: z,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_fg,lane_mIoU,
#            train_wall_min,peak_gpu_mem_mib,final_train_loss,seed,git_commit
#
# Each (z, epoch) -> experiments/phase2d/expD_z{Z}_e{EP}/ + _eval/
#
# Control:
#   - STOP : touch experiments/phase2d/STOP
#   - RESUME: cell skipped if _eval/metrics.json exists
#   - APPEND CSV, never truncate
#
# Usage:
#   run_in_background=true wsl.exe -e bash -lc 'cd /home/mycode/ai_study/trac && bash scripts/phase2d_run_budget.sh 2>&1 | tee experiments/phase2d/expD_foreground.log'
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

ZS=(${ZS_OVERRIDE:-16 32 128})
EPS=(${EPS_OVERRIDE:-10 20})
SEED=${SEED_OVERRIDE:-0}
OUT=experiments/phase2d
mkdir -p "$OUT"
CSV="$OUT/expD_budget.csv"
RUNNER_LOG="$OUT/expD_runner.log"
STOP_FILE="$OUT/STOP"
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

HEADER="z,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_fg,lane_mIoU,train_wall_min,peak_gpu_mem_mib,final_train_loss,seed,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

echo "===== [$(date '+%F %T')] PHASE 2-D BUDGET RUNNER (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
echo "zs=${ZS[*]}  eps=${EPS[*]}  seed=${SEED}" | tee -a "$RUNNER_LOG"

# Helper: extract peak GPU mem and final loss from training_log.txt
extract_train_stats() {
  local LOG="$1"
  local PEAK_MEM="NA"
  local FINAL_LOSS="NA"
  if [ -f "$LOG" ]; then
    PEAK_MEM=$(grep -oE 'mem [0-9]+/[0-9]+MiB' "$LOG" | awk '{print $2}' | sort -t/ -k1 -n -r | head -1 | tr -d '/' | sed 's/MiB$//')
    if [ -z "$PEAK_MEM" ]; then
      PEAK_MEM=$(awk '{for(i=1;i<=NF;i++) if($i=="mem") print $(i+1)}' "$LOG" | awk -F'/' '{print $1}' | tr -d 'M' | sort -n | tail -1)
    fi
    FINAL_LOSS=$(grep -oE 'avg_loss=[0-9.]+' "$LOG" | tail -1 | sed 's/avg_loss=//')
  fi
  echo "${PEAK_MEM} ${FINAL_LOSS}"
}

for Z in "${ZS[@]}"; do
  for EP in "${EPS[@]}"; do
    if [ -f "$STOP_FILE" ]; then
      echo "===== [$(date '+%F %T')] STOP present — halt before z=${Z} e=${EP} =====" | tee -a "$RUNNER_LOG"
      exit 0
    fi
    CFG="configs/phase2b_zsweep_z${Z}.yaml"
    if [ ! -f "$CFG" ]; then
      echo "[WARN] missing config $CFG — skip z=${Z}" | tee -a "$RUNNER_LOG"
      continue
    fi
    TAG="expD_z${Z}_e${EP}"
    TR="$OUT/$TAG"
    EVL="$OUT/${TAG}_eval"
    CKPT="$TR/checkpoint.pt"
    MP="$EVL/metrics.json"
    TLOG="$TR/training_log.txt"

    if [ -f "$MP" ]; then
      echo "[resume] ${TAG} already evaluated — skip" | tee -a "$RUNNER_LOG"
      continue
    fi

    mkdir -p "$TR" "$EVL"

    # --- checkpoint integrity check -------------------------------------
    # train.py overwrites checkpoint.pt every epoch, so an interrupted run
    # leaves a PARTIAL model (e.g. epoch 8 of a 10ep job). Evaluating that
    # as a 10ep result would silently corrupt the dataset, so verify the
    # stored epoch count matches the target before trusting it.
    CK_EP=""
    if [ -f "$CKPT" ]; then
      CK_EP=$("$PY" -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch','ERR'))" "$CKPT" 2>/dev/null)
    fi
    if [ -f "$CKPT" ] && [ "$CK_EP" = "$EP" ]; then
      echo "[resume] ${TAG} ckpt epoch=${CK_EP} matches target — eval only" | tee -a "$RUNNER_LOG"
    else
      if [ -f "$CKPT" ]; then
        echo "[resume] ${TAG} ckpt epoch=${CK_EP} != target ${EP} — DISCARD partial, retrain" | tee -a "$RUNNER_LOG"
        mv "$CKPT" "${CKPT}.partial_ep${CK_EP}"
      fi
      T0=$(date +%s)
      echo "===== [$(date '+%F %T')] z=${Z} epochs=${EP} seed=${SEED} TRAIN start =====" | tee -a "$RUNNER_LOG"
      "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs "$EP" --seed "$SEED" \
        2>&1 | tee -a "$TLOG" > "$TR/train_stdout.log"
      T1=$(date +%s)
      WALL_MIN=$(( (T1 - T0) / 60 ))
      echo "===== [$(date '+%F %T')] z=${Z} epochs=${EP} TRAIN done in ${WALL_MIN} min =====" | tee -a "$RUNNER_LOG"
    fi

    if [ ! -f "$CKPT" ]; then
      echo "[WARN] ${TAG} no checkpoint — NaN row" | tee -a "$RUNNER_LOG"
      echo "${Z},${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},${COMMIT}" >> "$CSV"
      continue
    fi

    echo "===== [$(date '+%F %T')] z=${Z} epochs=${EP} EVAL =====" | tee -a "$RUNNER_LOG"
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" \
      2>&1 | tee -a "$EVL.log" > "$EVL/eval_stdout.log"

    if [ ! -f "$MP" ]; then
      echo "[WARN] ${TAG} eval metrics missing — NaN row" | tee -a "$RUNNER_LOG"
      echo "${Z},${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},${COMMIT}" >> "$CSV"
      continue
    fi

    read -r PEAK_MEM FINAL_LOSS <<< "$(extract_train_stats "$TLOG")"
    # Also try to read wall time from runner log
    WALL_MIN=$(grep "TRAIN done in" "$RUNNER_LOG" | grep "z=${Z} epochs=${EP} " | tail -1 | grep -oE 'in [0-9]+ min' | grep -oE '[0-9]+')
    [ -z "$WALL_MIN" ] && WALL_MIN="NA"

    "$PY" - "$Z" "$EP" "$SEED" "$CKPT" "$COMMIT" "$CSV" "$MP" "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
z, ep, seed, ckpt, commit, csv, mp, wall, mem, floss = sys.argv[1:11]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
row = [str(z), str(ep),
       f"{g('parameters')/1e6:.3f}", f"{g('flops')/1e9:.3f}", f"{g('fps'):.2f}",
       f"{g('mAP50'):.4f}", f"{g('mAP50_95'):.4f}",
       f"{g('da_mIoU'):.4f}", f"{g('da_fg_iou'):.4f}",
       f"{g('lane_fg_iou'):.4f}", f"{g('lane_mIoU'):.4f}",
       str(wall), str(mem), str(floss), str(seed), commit]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF
  done
done

if [ -f "$STOP_FILE" ]; then
  echo "===== [$(date '+%F %T')] HALTED BY STOP =====" | tee -a "$RUNNER_LOG"
  exit 0
fi
echo "===== [$(date '+%F %T')] PHASE 2-D BUDGET RUNNER DONE =====" | tee -a "$RUNNER_LOG"
