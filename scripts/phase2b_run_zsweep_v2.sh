#!/bin/bash
# Phase-2-B Task 2(a) Z-capacity sweep runner — v2
#
# Differences vs v1 (scripts/phase2b_run_zsweep.sh):
#   1. GRACEFUL STOP: drop a file at experiments/phase2b/STOP to make the runner
#      exit between Z points (checked at the top of every loop iteration).
#          touch experiments/phase2b/STOP      # stops before the next Z
#          rm   experiments/phase2b/STOP       # re-enable
#   2. RESUME: a Z point is skipped when zsweep_z<N>_eval/metrics.json already
#      exists. If only checkpoint.pt exists (train done, eval missing), the run
#      skips training and goes straight to eval. So an interrupted sweep can be
#      restarted with the exact same command and will not redo finished work.
#   3. Z-LIST OVERRIDE: run a subset without editing the script.
#          ZS_OVERRIDE="32 48 64" bash scripts/phase2b_run_zsweep_v2.sh
#   4. APPENDS to zsweep_results.csv instead of truncating it (v1 wiped it at
#      startup, which destroys partial results if you restart mid-sweep).
#
# IMPORTANT: do NOT edit scripts/phase2b_run_zsweep.sh while v1 is running —
# bash has already parsed its loop into memory, and rewriting the file in place
# can make it execute misaligned bytes when it next reads from the file.
# Use this v2 script for any (re)start.
#
# Launch:
#   cd /home/mycode/ai_study/trac
#   setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
#     > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

ZS=(${ZS_OVERRIDE:-"16 32 48 64 96 128"})
OUT=experiments/phase2b
mkdir -p "$OUT"
CSV="$OUT/zsweep_results.csv"
RUNNER_LOG="$OUT/zsweep_runner.log"
STOP_FILE="$OUT/STOP"

HEADER="z,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_fg,lane_mIoU,checkpoint,seed,epochs,git_commit"
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"

emit_row() {
  "$PY" - "$1" "$2" "$3" "$CSV" "$4" <<'PYEOF'
import json, sys
z, ckpt, commit, csv, mp = sys.argv[1:6]
m = json.load(open(mp))
def g(k, d=0.0):
    return m.get(k, d)
row = [z,
       f"{g('parameters')/1e6:.3f}", f"{g('flops')/1e9:.3f}", f"{g('fps'):.2f}",
       f"{g('mAP50'):.4f}", f"{g('mAP50_95'):.4f}",
       f"{g('da_mIoU'):.4f}", f"{g('da_fg_iou'):.4f}",
       f"{g('lane_fg_iou'):.4f}", f"{g('lane_mIoU'):.4f}",
       ckpt, "0", "4", commit]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF
}

for Z in "${ZS[@]}"; do
  if [ -f "$STOP_FILE" ]; then
    echo "===== [$(date '+%F %T')] STOP FILE PRESENT — halting before Z=${Z} =====" | tee -a "$RUNNER_LOG"
    echo "Remove $STOP_FILE to continue." | tee -a "$RUNNER_LOG"
    exit 0
  fi

  CFG="configs/phase2b_zsweep_z${Z}.yaml"
  TR="$OUT/zsweep_z${Z}"
  EVL="$OUT/zsweep_z${Z}_eval"
  CKPT="$TR/checkpoint.pt"
  MP="$EVL/metrics.json"
  COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

  if [ -f "$MP" ]; then
    echo "===== [$(date '+%F %T')] Z=${Z} already evaluated — SKIP (resume) =====" | tee -a "$RUNNER_LOG"
    continue
  fi

  echo "===== [$(date '+%F %T')] Z=${Z} START (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"

  if [ -f "$CKPT" ]; then
    echo "[resume] checkpoint exists, skipping training for Z=${Z}" | tee -a "$RUNNER_LOG"
  else
    echo "----- Z=${Z} TRAIN -----" | tee -a "$RUNNER_LOG"
    "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs 4 --seed 0 2>&1 | tee -a "$TR.log"
  fi

  if [ ! -f "$CKPT" ]; then
    echo "[WARN] Z=${Z} produced no checkpoint — recording NaN row." | tee -a "$RUNNER_LOG"
    echo "${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},0,4,${COMMIT}" >> "$CSV"
    continue
  fi

  echo "----- Z=${Z} EVAL -----" | tee -a "$RUNNER_LOG"
  "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" 2>&1 | tee -a "$EVL.log"

  if [ ! -f "$MP" ]; then
    echo "[WARN] Z=${Z} eval metrics missing — recording NaN row." | tee -a "$RUNNER_LOG"
    echo "${Z},NA,NA,NA,NA,NA,NA,NA,NA,NA,${CKPT},0,4,${COMMIT}" >> "$CSV"
    continue
  fi

  emit_row "$Z" "$CKPT" "$COMMIT" "$MP" | tee -a "$RUNNER_LOG"
done

if [ -f "$STOP_FILE" ]; then
  echo "===== [$(date '+%F %T')] HALTED BY STOP FILE =====" | tee -a "$RUNNER_LOG"
  exit 0
fi

echo "===== [$(date '+%F %T')] Z-SWEEP DONE =====" | tee -a "$RUNNER_LOG"
"$PY" scripts/phase2b_analyze.py --root "$ROOT" --mode auto \
  --out "$OUT/zsweep_analysis.md" 2>&1 | tee -a "$RUNNER_LOG"
