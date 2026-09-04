#!/bin/bash
# Phase 2-C · Experiment A — confirm Z=16/32/128 with extra seeds.
#
# Decision-gate spec: keep z in {16,32,128}; seed0 already exists (zsweep).
# This runner adds seed1 and seed2 for each z, using the EXACT fixed protocol
# (4ep / bs16 / AdamW lr1e-3 / cosine / tri_train / same encoder).
#
# Output: experiments/phase2c/expA_multiseed.csv  (independent of zsweep_results.csv)
# Each (z,seed) -> experiments/phase2c/expA_z<Z>_s<SEED>/checkpoint.pt + _eval/metrics.json
#
# Control (mirrors zsweep runner):
#   - GRACEFUL STOP: touch experiments/phase2c/STOP  (checked between units)
#   - RESUME: a (z,seed) is skipped if its _eval/metrics.json exists; only
#     checkpoint -> go straight to eval.
#   - APPEND CSV, never truncate.
#
# Usage:
#   setsid nohup bash scripts/phase2c_run_expA.sh > experiments/phase2c/expA_stdout.log 2>&1 < /dev/null & disown
#
# Optional: SEEDS_OVERRIDE="1" ZS_OVERRIDE="16 128" to run a subset.
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

ZS=(${ZS_OVERRIDE:-16 32 128})
SEEDS=(${SEEDS_OVERRIDE:-1 2})
OUT=experiments/phase2c
mkdir -p "$OUT"
CSV="$OUT/expA_multiseed.csv"
RUNNER_LOG="$OUT/expA_runner.log"
STOP_FILE="$OUT/STOP"

HEADER="z,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_fg,lane_mIoU,checkpoint,seed,epochs,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

for Z in "${ZS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    if [ -f "$STOP_FILE" ]; then
      echo "===== [$(date '+%F %T')] STOP present — halt before z=${Z} s=${SEED} =====" | tee -a "$RUNNER_LOG"
      exit 0
    fi
    CFG="configs/phase2b_zsweep_z${Z}.yaml"
    if [ ! -f "$CFG" ]; then
      echo "[WARN] missing config $CFG — skip z=${Z}" | tee -a "$RUNNER_LOG"
      continue
    fi
    TAG="expA_z${Z}_s${SEED}"
    TR="$OUT/$TAG"
    EVL="$OUT/${TAG}_eval"
    CKPT="$TR/checkpoint.pt"
    MP="$EVL/metrics.json"
    COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

    if [ -f "$MP" ]; then
      echo "[resume] ${TAG} already evaluated — skip" | tee -a "$RUNNER_LOG"
      continue
    fi

    if [ -f "$CKPT" ]; then
      echo "[resume] ${TAG} ckpt exists — eval only" | tee -a "$RUNNER_LOG"
    else
      echo "===== [$(date '+%F %T')] z=${Z} seed=${SEED} TRAIN (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
      "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs 4 --seed "$SEED" 2>&1 | tee -a "$TR.log"
    fi

    if [ ! -f "$CKPT" ]; then
      echo "[WARN] ${TAG} no checkpoint — NaN row" | tee -a "$RUNNER_LOG"
      echo "${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},${SEED},4,${COMMIT}" >> "$CSV"
      continue
    fi

    echo "===== [$(date '+%F %T')] z=${Z} seed=${SEED} EVAL =====" | tee -a "$RUNNER_LOG"
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" 2>&1 | tee -a "$EVL.log"

    if [ ! -f "$MP" ]; then
      echo "[WARN] ${TAG} eval metrics missing — NaN row" | tee -a "$RUNNER_LOG"
      echo "${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},${SEED},4,${COMMIT}" >> "$CSV"
      continue
    fi

    "$PY" - "$Z" "$SEED" "$CKPT" "$COMMIT" "$CSV" "$MP" <<'PYEOF'
import json, sys
z, seed, ckpt, commit, csv, mp = sys.argv[1:7]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
row = [z,
       f"{g('parameters')/1e6:.3f}", f"{g('flops')/1e9:.3f}", f"{g('fps'):.2f}",
       f"{g('mAP50'):.4f}", f"{g('mAP50_95'):.4f}",
       f"{g('da_mIoU'):.4f}", f"{g('da_fg_iou'):.4f}",
       f"{g('lane_fg_iou'):.4f}", f"{g('lane_mIoU'):.4f}",
       ckpt, seed, "4", commit]
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
echo "===== [$(date '+%F %T')] EXP-A DONE =====" | tee -a "$RUNNER_LOG"
