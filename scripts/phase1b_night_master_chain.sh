#!/bin/bash
cd /home/mycode/ai_study/trac
log() { echo "[$(date '+%F %T')] $*"; }
waitfor() { for i in $(seq 1 1500); do [ -f "$1" ] && return 0; sleep 20; done; log "TIMEOUT $1"; return 1; }
log "START master chain"
rm -f experiments/phase4a/STOP_B
log "P4B5-STEP1 danc seed1 20ep"
CELLS_OVERRIDE="danc_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=1 CSV_OVERRIDE=experiments/phase5/phase5_det_e20_results.csv bash scripts/phase4b_run.sh 2>&1 | tail -8
waitfor experiments/phase4a/exp4B_danc_z16_e20_s1_eval/metrics.json || exit 1
log "seed1 done $(date +%T)"
sleep 90
rm -f experiments/phase4a/STOP_B
log "P4B5-STEP2 danc seed2 20ep"
CELLS_OVERRIDE="danc_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=2 CSV_OVERRIDE=experiments/phase5/phase5_det_e20_results.csv bash scripts/phase4b_run.sh 2>&1 | tail -8
waitfor experiments/phase4a/exp4B_danc_z16_e20_s2_eval/metrics.json || exit 1
log "seed2 done $(date +%T)"
sleep 90
rm -f experiments/phase4a/STOP_B
log "P4B5-STEP3 lane8 20ep"
CELLS_OVERRIDE="lane8_z16" EPOCHS_OVERRIDE=20 CSV_OVERRIDE=experiments/phase5/phase5_lane8_e20.csv bash scripts/phase4b_run.sh 2>&1 | tail -8
waitfor experiments/phase4a/exp4B_lane8_z16_e20_eval/metrics.json || exit 1
log "lane8 done $(date +%T)"
log "MASTER CHAIN COMPLETE at $(date +%T)"
