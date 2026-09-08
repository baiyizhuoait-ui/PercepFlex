#!/bin/bash
# Clean master chain (Plan B+): one training at a time, each via phase4b_run.sh
# which trains+eval+appends a CSV row synchronously. Survives across turns
# because the Bash-tool background task keeps this wsl.exe alive.
# NOTE: DataLoader num_workers child processes show full 'train.py' cmdlines;
# they are NOT separate jobs. Never kill them. Only ever kill the run.sh PID.
set -u
cd /home/mycode/ai_study/trac
RUN="bash scripts/phase4b_run.sh"
LOG=experiments/phase4a/clean_master_chain.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "START clean master chain (Plan B+)  seed plan: danc s1/s2 20ep, dp2b 4ep, lane8 20ep, da14 4ep"

# STEP1: danc seed1 20ep -> e20 csv (k-means default validation, cross-seed)
log "STEP1 danc seed1 20ep"
CELLS_OVERRIDE="danc_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=1 \
  CSV_OVERRIDE=experiments/phase5/phase5_det_e20_results.csv $RUN 2>&1 | tail -4
log "STEP1 done $(date +%T)"

# STEP2: danc seed2 20ep -> e20 csv
log "STEP2 danc seed2 20ep"
CELLS_OVERRIDE="danc_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=2 \
  CSV_OVERRIDE=experiments/phase5/phase5_det_e20_results.csv $RUN 2>&1 | tail -4
log "STEP2 done $(date +%T)"

# STEP3: dp2b 4ep -> probe csv (D3b vs danc 4ep, H21 closure)
log "STEP3 dp2b 4ep"
CELLS_OVERRIDE="dp2b_z16" EPOCHS_OVERRIDE=4 SEED_OVERRIDE=0 \
  CSV_OVERRIDE=experiments/phase5/phase5_det_probe_results.csv $RUN 2>&1 | tail -4
log "STEP3 done $(date +%T)"

# STEP4: lane8 20ep -> lane8 csv (4B-5 lane label widen 8px train / 2px eval)
log "STEP4 lane8 20ep"
CELLS_OVERRIDE="lane8_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=0 \
  CSV_OVERRIDE=experiments/phase5/phase5_lane8_e20.csv $RUN 2>&1 | tail -4
log "STEP4 done $(date +%T)"

# STEP5: da14 4ep -> da14 csv (4B-6 DA head 1/4 + f1 skip, new code, validated CPU fwd/bwd)
log "STEP5 da14 4ep"
CELLS_OVERRIDE="da14_z16" EPOCHS_OVERRIDE=4 SEED_OVERRIDE=0 \
  CSV_OVERRIDE=experiments/phase5/phase5_da14_results.csv $RUN 2>&1 | tail -4
log "STEP5 done $(date +%T)"

# STEP6: verdicts
log "STEP6 verdicts"
/home/mycode/ai_study/gpu_env/bin/python scripts/phase5_det_decide.py \
  --csv experiments/phase5/phase5_det_probe_results.csv \
  --out experiments/phase5/phase5_det_dp2b_decision.txt
/home/mycode/ai_study/gpu_env/bin/python scripts/phase5_det_decide.py \
  --csv experiments/phase5/phase5_det_e20_results.csv --baseline-cell r2_z16 \
  --out experiments/phase5/phase5_det_e20_decision.txt

log "CLEAN MASTER CHAIN COMPLETE at $(date '+%F %T')"
