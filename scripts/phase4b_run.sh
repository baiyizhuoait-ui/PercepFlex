#!/bin/bash
# Phase 4A Level 2, probe A - Task-Specific Projection Probe @ 4 epochs.
#
# Question: does giving each task its own projection off the shared Z help, and
# does it decouple task capacity from the shared-Z width?
#
# Four cells, E-base, seed 0, 4 epochs, 640x640, batch 16:
#   r2u_z16    R2 uniform  Z=16   (REUSED from Phase 4A STEP 2 sanity)
#   r2u_z32    R2 uniform  Z=32
#   r3tp_z16   R3 taskproj Z=16   det32 / lane32 / da16
#   r3tp_z32   R3 taskproj Z=32   det32 / lane32 / da16
#
# R3 target widths are absolute and identical across the two arms, so under R3
# every head sees the same input width regardless of the shared Z width. If the
# R3 z-gap collapses relative to the R2 z-gap, then what R2 was really paying
# for at wide z was head input width, not shared representational capacity.
#
# Usage:
#   bash scripts/phase4b_run.sh                       # all three new cells
#   CELLS_OVERRIDE="r3tp_z16" bash scripts/phase4b_run.sh
#
# Control:  touch experiments/phase4a/STOP_B to halt before the next cell.
set -u

PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

SEED=${SEED_OVERRIDE:-0}
EP=${EPOCHS_OVERRIDE:-4}

CELLS=(${CELLS_OVERRIDE:-r2u_z32 r3tp_z16 r3tp_z32})

OUT=experiments/phase4a
mkdir -p "$OUT"
CSV=${CSV_OVERRIDE:-$OUT/phase4B_probeA_results.csv}
RUNNER_LOG="$OUT/exp4B_runner.log"
STOP_FILE="$OUT/STOP_B"
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

HEADER="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
if [ ! -f "$CSV" ]; then
  echo "$HEADER" > "$CSV"
  # r2u_z16 is not retrained: its 4ep run is Phase 4A STEP 2 sanity, and the
  # model builds to identical parameters with task_proj defaulted off.
  "$PY" - "$CSV" <<'PYEOF'
import sys, csv, os
csv_path = sys.argv[1]
src = "experiments/phase4a/phase4A_sanity.csv"
if os.path.exists(src):
    rows = list(csv.DictReader(open(src)))
    r = [x for x in rows if x["cell"] == "r2_z16" and x["epochs"] == "4"]
    if r:
        r = dict(r[0])
        r["cell"] = "r2u_z16"
        r["variant"] = "r2"
        r["source"] = "reused:phase4a-sanity"
        fields = open(csv_path).readline().strip().split(",")
        with open(csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(r)
        print("REUSED r2u_z16 from Phase 4A STEP 2 sanity:", r["mAP50"])
    else:
        print("[WARN] no 4ep r2_z16 row in", src)
else:
    print("[WARN] missing", src)
PYEOF
fi

echo "===== [$(date '+%F %T')] PHASE 4B PROBE A (commit ${COMMIT}) =====" | tee -a "$RUNNER_LOG"
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

PROBE="$OUT/.ckpt_probe_B.py"
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
    echo "===== [$(date '+%F %T')] STOP_B present - halt before ${CELL} =====" | tee -a "$RUNNER_LOG"
    exit 0
  fi

  VAR="r2"; case "$CELL" in r3tp_*) VAR="r3";; esac
  Z="${CELL##*_z}"

  CFG="configs/phase4b_${CELL}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[WARN] missing config $CFG - skip ${CELL}" | tee -a "$RUNNER_LOG"
    continue
  fi

  if [ "$SEED" != "0" ]; then TAG="exp4B_${CELL}_e${EP}_s${SEED}"; else TAG="exp4B_${CELL}_e${EP}"; fi
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

  "$PY" - "$VAR" "$CELL" "$Z" "$EP" "$SEED" "$COMMIT" "$CSV" "$MP" \
        "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
(var, cell, z, ep, seed, commit, csv, mp, wall, mem, floss) = sys.argv[1:12]
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
  echo "===== [$(date '+%F %T')] HALTED BY STOP_B =====" | tee -a "$RUNNER_LOG"
  exit 0
fi
echo "===== [$(date '+%F %T')] PHASE 4B PROBE A RUNNER DONE =====" | tee -a "$RUNNER_LOG"
