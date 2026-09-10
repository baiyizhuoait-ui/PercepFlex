#!/bin/bash
# Phase 6B chain (P6B-STEP1): EXP-07 factorial screening (4 x 4ep, seed 0)
# then EXP-08 A-uniform 20ep seed0. NO 20ep EXP-07 confirmation here - that is
# gated on manual interaction analysis of the screening results.
# Halt politely any time: touch experiments/phase6/STOP_CHAIN
set -u
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1
FLAG=experiments/phase6/STOP_CHAIN
rm -f "$FLAG"
CSV_F=experiments/phase6/phase6_round2_factorial.csv
CSV_E=experiments/phase6/phase6_equal_budget.csv
HDR="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV_F" ] || echo "$HDR" > "$CSV_F"
[ -f "$CSV_E" ] || echo "$HDR" > "$CSV_E"

run () { # tag cfg ep seed csv cell z
  if [ -f "$FLAG" ]; then echo "[chain] STOP_CHAIN set - halt before $1"; exit 0; fi
  echo "[chain] $(date '+%F %T') -> $1"
  bash scripts/phase6_run.sh "$1" "$2" "$3" "$4" "$5" r2 "$6" "$7"
}

echo "[chain] P6B-STEP1 start $(date '+%F %T')"
# --- EXP-07 factorial screening (capacity x assignment), seed 0 ---
run e7_z16_old configs/phase4a_r2_z16.yaml    4 0 "$CSV_F" z16_old 16
run e7_z16_km  configs/phase4b_danc_z16.yaml  4 0 "$CSV_F" z16_km  16
run e7_z32_old configs/phase4a_r2_z32.yaml    4 0 "$CSV_F" z32_old 32
run e7_z32_km  configs/phase6_e7_z32_km.yaml  4 0 "$CSV_F" z32_km  32
# --- EXP-08 A-uniform (encoder width expansion @ ~1.64G), 20ep seed 0 ---
run e8_unif20  configs/phase6_e8_uniform_z16.yaml 20 0 "$CSV_E" e8_unif_seed0 16
echo "[chain] P6B-STEP1 DONE $(date '+%F %T')"
