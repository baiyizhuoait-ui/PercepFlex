#!/bin/bash
# Phase 3A · Encoder-Capacity Sanity Check.
#
# Question: is performance limited more by ENCODER capacity than by Z width?
#
# Fixed (identical to the Phase 2-D protocol):
#   z = 16, epochs = 20, batch = 16, lr = 1e-3, seed = 0,
#   same data / augment / optimizer / loss / input size / train script.
# Only the ENCODER WIDTH changes. Depth (blocks=[2,2,2]), Z width, and all
# three task heads are untouched.
#
#   E-small  stem=12 stages=[24,40,64,80]     ~0.101M params  0.722 GFLOPs
#   E-base   stem=16 stages=[32,64,96,128]    ~0.189M params  1.060 GFLOPs
#   E-large  stem=20 stages=[40,80,128,168]   ~0.292M params  1.428 GFLOPs
#
# E-base REUSE: an identical run already exists from Phase 2-D
# (experiments/phase2d/expD_z16_e20: same encoder, z=16, 20ep, seed=0).
# Its checkpoint / training log / eval metrics are symlinked in rather than
# re-training, saving ~132 min of GPU time. The reuse is flagged in the CSV
# (source column) and in the report.
#
# Output: experiments/phase3a/exp3A_encoder.csv
#
# Control:
#   STOP   : touch experiments/phase3a/STOP
#   RESUME : a cell whose _eval/metrics.json exists is NOT re-trained; its
#            metrics are read and appended (so reused cells still get a row).
#   CHECKPOINT INTEGRITY: a checkpoint is trusted only if its stored epoch
#            equals the target epoch count; partial files are renamed aside.
#
# Usage:
#   run_in_background=true wsl.exe -e bash -lc \
#     'cd /home/mycode/ai_study/trac && bash scripts/phase3a_run_encoder.sh 2>&1 | tee experiments/phase3a/exp3A_foreground.log'
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

CELLS=(${CELLS_OVERRIDE:-esmall ebase elarge})
SEED=${SEED_OVERRIDE:-0}
EP=${EPOCHS_OVERRIDE:-20}
OUT=experiments/phase3a
mkdir -p "$OUT"
CSV="$OUT/exp3A_encoder.csv"
RUNNER_LOG="$OUT/exp3A_runner.log"
STOP_FILE="$OUT/STOP"
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

HEADER="encoder,stem,stages,z,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

echo "===== [$(date '+%F %T')] PHASE 3A ENCODER RUNNER (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
echo "cells=${CELLS[*]}  epochs=${EP}  seed=${SEED}" | tee -a "$RUNNER_LOG"

# ---- peak GPU mem + final train loss from training_log.txt -------------------
extract_train_stats() {
  local LOG="$1"
  local PEAK_MEM="NA"
  local FINAL_LOSS="NA"
  if [ -f "$LOG" ]; then
    # Log lines look like: "... gpu%  82 mem 2700/8151MiB | loss ..."
    # Take the NUMERATOR (MiB used) BEFORE any slash removal, otherwise
    # "2700/8151MiB" becomes "27008151" (numerator glued to denominator).
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

# ---- E-base: reuse the Phase 2-D run instead of re-training ------------------
setup_reuse() {
  local TR="$1" EVL="$2"
  local SRC_TR="experiments/phase2d/expD_z16_e20"
  local SRC_EV="experiments/phase2d/expD_z16_e20_eval"
  if [ ! -f "$SRC_TR/checkpoint.pt" ]; then
    echo "[reuse] source $SRC_TR/checkpoint.pt missing — will train E-base from scratch" | tee -a "$RUNNER_LOG"
    return 1
  fi
  mkdir -p "$TR" "$EVL"
  ln -sf "$ROOT/$SRC_TR/checkpoint.pt"     "$TR/checkpoint.pt"
  ln -sf "$ROOT/$SRC_TR/training_log.txt"  "$TR/training_log.txt"
  ln -sf "$ROOT/$SRC_EV/metrics.json"      "$EVL/metrics.json"
  echo "[reuse] E-base linked to $SRC_TR (identical config+protocol)" | tee -a "$RUNNER_LOG"
  return 0
}

for CELL in "${CELLS[@]}"; do
  if [ -f "$STOP_FILE" ]; then
    echo "===== [$(date '+%F %T')] STOP present — halt before ${CELL} =====" | tee -a "$RUNNER_LOG"
    exit 0
  fi
  CFG="configs/phase3a_${CELL}_z16.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[WARN] missing config $CFG — skip ${CELL}" | tee -a "$RUNNER_LOG"
    continue
  fi
  TAG="exp3A_${CELL}_z16"
  TR="$OUT/$TAG"
  EVL="$OUT/${TAG}_eval"
  CKPT="$TR/checkpoint.pt"
  MP="$EVL/metrics.json"
  TLOG="$TR/training_log.txt"
  SRC="trained"

  mkdir -p "$TR" "$EVL"

  if [ "$CELL" = "ebase" ] && [ ! -f "$MP" ]; then
    if setup_reuse "$TR" "$EVL"; then SRC="reused:phase2d/expD_z16_e20"; fi
  fi

  # ---- checkpoint integrity check -----------------------------------------
  CK_EP=""
  if [ -f "$CKPT" ] && [ ! -L "$CKPT" ]; then
    CK_EP=$("$PY" -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch','ERR'))" "$CKPT" 2>/dev/null)
  elif [ -L "$CKPT" ]; then
    CK_EP=$("$PY" -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('epoch','ERR'))" "$CKPT" 2>/dev/null)
  fi

  if [ -f "$MP" ]; then
    echo "[resume] ${TAG} metrics present — append without retraining" | tee -a "$RUNNER_LOG"
  elif [ -f "$CKPT" ] && [ "$CK_EP" = "$EP" ]; then
    echo "[resume] ${TAG} ckpt epoch=${CK_EP} matches target — eval only" | tee -a "$RUNNER_LOG"
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" \
      2>&1 | tee -a "$EVL.log" > "$EVL/eval_stdout.log"
  else
    if [ -f "$CKPT" ] && [ ! -L "$CKPT" ] && [ "$CK_EP" != "$EP" ]; then
      echo "[resume] ${TAG} ckpt epoch=${CK_EP} != target ${EP} — DISCARD partial, retrain" | tee -a "$RUNNER_LOG"
      mv "$CKPT" "${CKPT}.partial_ep${CK_EP}"
    fi
    T0=$(date +%s)
    echo "===== [$(date '+%F %T')] ${CELL} epochs=${EP} seed=${SEED} TRAIN start =====" | tee -a "$RUNNER_LOG"
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
    echo "[WARN] ${TAG} no metrics — NaN row" | tee -a "$RUNNER_LOG"
    echo "${CELL},NA,NA,16,${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},${SRC},${COMMIT}" >> "$CSV"
    continue
  fi

  read -r PEAK_MEM FINAL_LOSS <<< "$(extract_train_stats "$TLOG")"
  WALL_MIN=$(grep "TRAIN done in" "$RUNNER_LOG" | grep "${CELL} " | tail -1 | grep -oE 'in [0-9]+ min' | grep -oE '[0-9]+')
  [ -z "$WALL_MIN" ] && WALL_MIN="NA"

  STEM=$("$PY" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['model']['encoder']['stem'])" "$CFG")
  STAGES=$("$PY" -c "import yaml,sys;print('-'.join(str(c) for c in yaml.safe_load(open(sys.argv[1]))['model']['encoder']['stages']))" "$CFG")

  "$PY" - "$CELL" "$STEM" "$STAGES" "$EP" "$SEED" "$CKPT" "$COMMIT" "$CSV" "$MP" \
        "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" "$SRC" <<'PYEOF'
import json, sys
(cell, stem, stages, ep, seed, ckpt, commit, csv, mp,
 wall, mem, floss, src) = sys.argv[1:14]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
row = [cell, stem, stages, "16", ep,
       f"{g('parameters')/1e6:.4f}", f"{g('flops')/1e9:.4f}", f"{g('fps'):.2f}",
       f"{g('mAP50'):.4f}", f"{g('mAP50_95'):.4f}",
       f"{g('da_mIoU'):.4f}", f"{g('da_fg_iou'):.4f}",
       f"{g('lane_mIoU'):.4f}", f"{g('lane_fg_iou'):.4f}",
       str(mem), str(floss), str(wall), seed, src, commit]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF
done

if [ -f "$STOP_FILE" ]; then
  echo "===== [$(date '+%F %T')] HALTED BY STOP =====" | tee -a "$RUNNER_LOG"
  exit 0
fi
echo "===== [$(date '+%F %T')] PHASE 3A ENCODER RUNNER DONE =====" | tee -a "$RUNNER_LOG"
