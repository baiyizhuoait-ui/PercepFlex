#!/bin/bash
# Phase 6C / EXP-09 P6C-STEP1 chain -- capacity x anchor 2x2 factorial, missing cells.
#
# Preregistration: docs/PHASE6C_CAPACITY_ANCHOR_PREREGISTRATION.md  (READ IT FIRST)
#
# Cell map (20ep, seed0, ebase, batch16):
#   C1  r2_z16      x old anchors = experiments/phase4a/exp4A_r2_z16_e20          (0.3543)
#   C2  r2_z16      x k-means     = experiments/phase4a/exp4B_danc_z16_e20_s1/s2 (0.4982/0.4960)
#   C3  A-uniform   x old anchors = experiments/phase6/exp6_e8_unif20            (0.5339)
#   C4  A-uniform   x k-means     = <-- run 1 below (the decisive cell)
#
# Halt politely at ANY run boundary:  touch experiments/phase6c/STOP_CHAIN
# P6C-STEP2 (3-seed confirmation) is GATED on P6C-STEP1 reading SUBSTITUTION -- not here.
set -u
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1
OUT=experiments/phase6c
FLAG=$OUT/STOP_CHAIN
mkdir -p "$OUT"
rm -f "$FLAG"
CSV=$OUT/phase6c_e9_factorial.csv
HDR="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HDR" > "$CSV"

run () { # tag cfg ep seed cell z
  if [ -f "$FLAG" ]; then echo "[chain] STOP_CHAIN set - halt before $1"; exit 0; fi
  echo "[chain] $(date '+%F %T') -> $1"
  bash scripts/phase6_run.sh "$1" "$2" "$3" "$4" "$CSV" r2 "$5" "$6" "$OUT" exp9_
}

echo "[chain] PHASE6C EXP-09 P6C-STEP1 start $(date '+%F %T')"
# --- decisive cell first: A-uniform x k-means (pairs with exp6_e8_unif20) ---
run aunif_km configs/phase6c_e9_aunif_km.yaml 20 0 aunif_km 16
# --- seed-match the low-capacity pair (0.4982/0.4960 were s1/s2) ---
run r2z16_km configs/phase6c_e9_r2z16_km.yaml 20 0 r2z16_km 16
echo "[chain] PHASE6C EXP-09 P6C-STEP1 DONE $(date '+%F %T')"
