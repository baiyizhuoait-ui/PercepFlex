#!/usr/bin/env bash
# Wait for the running 20ep confirmation to finish, then run the zero-training
# error-geometry diagnostic. Deliberately serial: a diagnostic launched on the
# same GPU mid-training has already cost this project one OOM-killed run.
set -u
cd /home/mycode/ai_study/trac || exit 1
LOG=experiments/phase4a/exp4C_errorgeom.log
: > "$LOG"

log() { echo "===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }

log "waiting for phase4b_run.sh / train.py to exit"
while pgrep -f "phase4b_run.sh" > /dev/null || pgrep -f "training/train.py" > /dev/null; do
  sleep 30
done
log "GPU free - smoke test (2 images)"

PY=/home/mycode/ai_study/gpu_env/bin/python
"$PY" scripts/phase4c_errorgeom.py --n 2 --bs 1 >> "$LOG" 2>&1
RC=$?
log "smoke exit=$RC"

if [ "$RC" -ne 0 ]; then
  log "SMOKE FAILED - full run skipped, nothing to interpret"
  exit 1
fi

log "full run (300 val images x 3 cells)"
"$PY" scripts/phase4c_errorgeom.py --n 300 --bs 1 >> "$LOG" 2>&1
log "full exit=$?"
