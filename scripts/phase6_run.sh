#!/bin/bash
# Phase 6 generic single-run runner (train + eval + append row).
# Usage: bash scripts/phase6_run.sh <tag> <config> <epochs> <seed> <csv> <var> <cell> <z>
# Resume-safe: if metrics.json already exists for this outdir, eval-only reuse.
set -u
PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1

TAG="$1"; CFG="$2"; EP="$3"; SEED="$4"; CSV="$5"; VAR="$6"; CELL="$7"; Z="$8"
OUT="${9:-experiments/phase6}"
PFX="${10:-exp6_}"
TR="$OUT/${PFX}${TAG}"
EVL="${TR}_eval"
CKPT="$TR/checkpoint.pt"
MP="$EVL/metrics.json"
TLOG="$OUT/train_${TAG}.log"
mkdir -p "$OUT"

# skip if already appended (idempotent relaunch)
if [ -f "$CSV" ] && awk -F, -v c="$CELL" -v s="$SEED" 'NR>1 && $2==c && $18==s {f=1} END{exit !f}' "$CSV"; then
  echo "[phase6] ${CELL} seed${SEED} already in ${CSV} - SKIP"
  exit 0
fi

if [ -f "$MP" ]; then
  echo "[phase6] ${TAG} metrics present - eval reuse" | tee -a "$OUT/phase6_runner.log"
else
  mkdir -p "$TR" "$EVL"
  T0=$(date +%s)
  echo "===== [$(date '+%F %T')] ${TAG} TRAIN start (ep=${EP} seed=${SEED}) =====" | tee -a "$OUT/phase6_runner.log"
  "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs "$EP" --seed "$SEED" \
    2>&1 | tee "$TLOG" > "$TR/train_stdout.log"
  T1=$(date +%s)
  WALL_MIN=$(( (T1 - T0) / 60 ))
  echo "===== [$(date '+%F %T')] ${TAG} TRAIN done in ${WALL_MIN} min =====" | tee -a "$OUT/phase6_runner.log"
  if [ -f "$CKPT" ]; then
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" \
      2>&1 | tee -a "${EVL}.log" > "$EVL/eval_stdout.log"
  fi
fi

if [ ! -f "$MP" ]; then
  echo "${VAR},${CELL},${Z},ebase,${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},failed-no-metrics,$(git rev-parse HEAD 2>/dev/null || echo unknown)" >> "$CSV"
  echo "[FAIL] ${TAG} no metrics.json" | tee -a "$OUT/phase6_runner.log"
  exit 1
fi

WALL_MIN=${WALL_MIN:-NA}
PEAK_MEM=$(grep -aoE "peak_mem=[0-9]+" "$TLOG" 2>/dev/null | tail -n1 | cut -d= -f2)
PEAK_MEM=${PEAK_MEM:-NA}
FINAL_LOSS=$(grep -aoE "loss[^0-9]*[0-9]+\.[0-9]+" "$TLOG" 2>/dev/null | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1)
FINAL_LOSS=${FINAL_LOSS:-NA}

"$PY" - "$VAR" "$CELL" "$Z" "$EP" "$SEED" "$CSV" "$MP" "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
(var, cell, z, ep, seed, csv, mp, wall, mem, floss) = sys.argv[1:11]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
def num(v):
    try: f = float(v)
    except (TypeError, ValueError): return "NA"
    return "NA" if f != f else "%.4f" % f
row = [var, cell, z, "ebase", ep,
       num(g("parameters") / 1e6 if g("parameters") else "NA"),
       num(g("flops") / 1e9 if g("flops") else "NA"),
       "%.2f" % g("fps"), num(g("mAP50")), num(g("mAP50_95")),
       num(g("da_mIoU")), num(g("da_fg_iou")),
       num(g("lane_mIoU")), num(g("lane_fg_iou")),
       str(mem), str(floss), str(wall), seed, "trained",
       __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:40] or "unknown"]
with open(csv, "a") as f:
    f.write(",".join(row) + "\n")
print("APPENDED:", ",".join(row))
PYEOF
echo "[phase6] ${TAG} DONE" | tee -a "$OUT/phase6_runner.log"
