#!/bin/bash
# Clean master chain (Plan B+): one training at a time, each via phase4b_run.sh
# which trains+eval+appends a CSV row synchronously. Survives across turns
# because the Bash-tool background task keeps this wsl.exe alive.
#
# RESUME-SAFE: every step first checks whether its CSV row already exists
# (cell + seed). If so it is SKIPPED, so a killed chain can be relaunched
# without redoing completed steps or creating duplicate rows. Furthermore the
# underlying training (train.py + phase4b_run.sh) is now epoch-resume-safe: an
# interrupted job continues from its latest completed epoch, not from scratch.
#
# STOP SAFELY: to halt the chain, NEVER `kill` the master process (that kills
# the whole process group, including the running train.py). Instead:
#     touch experiments/phase4a/STOP_CHAIN
# The chain finishes the current step, then exits cleanly before the next one.
# To abort a running train.py, kill ONLY its PID; relaunch this chain and the
# resume logic picks up from the last completed epoch.
#
# NOTE: DataLoader num_workers child processes show full 'train.py' cmdlines;
# they are NOT separate jobs. Never kill them.
set -u
cd /home/mycode/ai_study/trac
RUN="bash scripts/phase4b_run.sh"
LOG=experiments/phase4a/clean_master_chain.log
STOP_FILE="experiments/phase4a/STOP_CHAIN"
maybe_stop() {
  if [ -f "$STOP_FILE" ]; then
    log "STOP_CHAIN present - halting now (no process killed). Remove it to resume."
    exit 0
  fi
}

# row_exists <csv> <cell> [seed]  -> exit 0 if a COMPLETED row is present.
# A failed/stale row (mAP50=NA or source=failed-no-metrics) does NOT count as
# done: the step must (re)run. Only rows with real metrics block re-execution.
row_exists() {
  local csv="$1" cell="$2" seed="${3:-}"
  [ -f "$csv" ] || return 1
  if [ -n "$seed" ]; then
    awk -F, -v c="$cell" -v s="$seed" 'NR>1 && $2==c && $18==s && $9!="NA" && $19!="failed-no-metrics" {f=1} END{exit !f}' "$csv"
  else
    awk -F, -v c="$cell" 'NR>1 && $2==c && $9!="NA" && $19!="failed-no-metrics" {f=1} END{exit !f}' "$csv"
  fi
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "START clean master chain (Plan B+, resume-safe)  seed plan: danc s1/s2 20ep, dp2b 4ep, lane8 20ep, da14 4ep"

# P4B5-STEP1: danc seed1 20ep -> e20 csv (k-means default validation, cross-seed)
maybe_stop
if row_exists experiments/phase5/phase5_det_e20_results.csv danc_z16 1; then
  log "P4B5-STEP1 SKIP (danc_z16 seed1 already in e20 csv)"
else
  log "P4B5-STEP1 danc seed1 20ep"
  CELLS_OVERRIDE="danc_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=1 \
    CSV_OVERRIDE=experiments/phase5/phase5_det_e20_results.csv $RUN 2>&1 | tail -4
  log "P4B5-STEP1 done $(date +%T)"
fi

# P4B5-STEP2: danc seed2 20ep -> e20 csv
maybe_stop
if row_exists experiments/phase5/phase5_det_e20_results.csv danc_z16 2; then
  log "P4B5-STEP2 SKIP (danc_z16 seed2 already in e20 csv)"
else
  log "P4B5-STEP2 danc seed2 20ep"
  CELLS_OVERRIDE="danc_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=2 \
    CSV_OVERRIDE=experiments/phase5/phase5_det_e20_results.csv $RUN 2>&1 | tail -4
  log "P4B5-STEP2 done $(date +%T)"
fi

# P4B5-STEP3: dp2b 4ep -> probe csv (D3b vs danc 4ep, H-21 closure)
maybe_stop
if row_exists experiments/phase5/phase5_det_probe_results.csv dp2b_z16; then
  log "P4B5-STEP3 SKIP (dp2b_z16 already in probe csv)"
else
  log "P4B5-STEP3 dp2b 4ep"
  CELLS_OVERRIDE="dp2b_z16" EPOCHS_OVERRIDE=4 SEED_OVERRIDE=0 \
    CSV_OVERRIDE=experiments/phase5/phase5_det_probe_results.csv $RUN 2>&1 | tail -4
  log "P4B5-STEP3 done $(date +%T)"
fi

# P4B5-STEP4: lane8 20ep -> lane8 csv (P4B-EXP-05 lane label widen 8px train / 2px eval)
maybe_stop
if row_exists experiments/phase5/phase5_lane8_e20.csv lane8_z16; then
  log "P4B5-STEP4 SKIP (lane8_z16 already in lane8 csv)"
else
  log "P4B5-STEP4 lane8 20ep"
  CELLS_OVERRIDE="lane8_z16" EPOCHS_OVERRIDE=20 SEED_OVERRIDE=0 \
    CSV_OVERRIDE=experiments/phase5/phase5_lane8_e20.csv $RUN 2>&1 | tail -4
  log "P4B5-STEP4 done $(date +%T)"
fi

# P4B5-STEP5: da14 4ep -> da14 csv (P4B-EXP-06 DA head 1/4 + f1 skip, new code, validated CPU fwd/bwd)
maybe_stop
if row_exists experiments/phase5/phase5_da14_results.csv da14_z16; then
  log "P4B5-STEP5 SKIP (da14_z16 already in da14 csv)"
else
  log "P4B5-STEP5 da14 4ep"
  CELLS_OVERRIDE="da14_z16" EPOCHS_OVERRIDE=4 SEED_OVERRIDE=0 \
    CSV_OVERRIDE=experiments/phase5/phase5_da14_results.csv $RUN 2>&1 | tail -4
  log "P4B5-STEP5 done $(date +%T)"
fi

# P5-STEP6: verdicts (seed-aware; reports every danc_z16 row)
log "P5-STEP6 verdicts"
/home/mycode/ai_study/gpu_env/bin/python scripts/phase5_det_decide.py \
  --csv experiments/phase5/phase5_det_probe_results.csv \
  --out experiments/phase5/phase5_det_dp2b_decision.txt
/home/mycode/ai_study/gpu_env/bin/python scripts/phase5_det_decide.py \
  --csv experiments/phase5/phase5_det_e20_results.csv --baseline-cell r2_z16 \
  --out experiments/phase5/phase5_det_e20_decision.txt

log "CLEAN MASTER CHAIN COMPLETE at $(date '+%F %T')"
