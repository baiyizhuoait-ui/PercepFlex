#!/bin/bash
# Chains the queued GPU work: wait for the running 20ep confirmation, then run
# probe B (3 cells @4ep), then probe C (2 cells @20ep), then the zero-training
# gradient / effective-rank diagnostic on the probe C checkpoints.
#
# Launched as its own background task so it does not depend on the shell that
# started the 20ep job still being alive.
set -u
PY=/home/mycode/ai_study/gpu_env/bin/python
ROOT=/home/mycode/ai_study/trac
cd "$ROOT" || exit 1
LOG=experiments/phase4a/exp4BC_chain.log

say() { echo "===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }

say "CHAIN START - waiting for the running r3tp_z16 20ep confirmation"

# Poll until nothing belonging to that run is alive. Cap the wait so a stuck job
# cannot hold the chain forever.
WAITED=0
while pgrep -f "exp4B_r3tp_z16_e20" > /dev/null 2>&1; do
  sleep 30
  WAITED=$((WAITED + 30))
  if [ "$WAITED" -ge 14400 ]; then
    say "WAIT TIMEOUT 4h - aborting chain"
    exit 1
  fi
done
sleep 20
say "20ep confirmation finished - probe B next"

rm -f experiments/phase4a/STOP_B
if [ -f experiments/phase4a/phase4B_probeB_results.csv ]; then
  say "probe B results csv already exists - skipping probe B"
else
  CELLS_OVERRIDE="r2d_z16 r2d_z32 r2p_z16" EPOCHS_OVERRIDE=4 \
    CSV_OVERRIDE=experiments/phase4a/phase4B_probeB_results.csv \
    bash scripts/phase4b_run.sh >> experiments/phase4a/exp4B_probeB_foreground.log 2>&1
  say "probe B exit=$?"
fi

say "probe C next"
rm -f experiments/phase4a/STOP_C
bash scripts/phase4c_rw_run.sh >> experiments/phase4a/exp4C_foreground.log 2>&1
say "probe C exit=$?"

# ---- zero-training diagnostic on the probe C checkpoints (manipulation check) ----
say "probe C gradient + effective-rank diagnostic"
"$PY" - <<'PYEOF' >> experiments/phase4a/exp4C_diag.log 2>&1
import re, pathlib
ROOT = pathlib.Path("/home/mycode/ai_study/trac")
src = (ROOT / "scripts/phase4a_diagnose.py").read_text()
cells = '''CELLS = [
    ("r2rw", "rw_z16", "configs/phase4c_rw_z16.yaml", "experiments/phase4a/exp4C_rw_z16_e20/checkpoint.pt"),
    ("r2rw", "rw_z32", "configs/phase4c_rw_z32.yaml", "experiments/phase4a/exp4C_rw_z32_e20/checkpoint.pt"),
]'''
src = re.sub(r"CELLS = \[.*?\n\]", cells, src, flags=re.S)
src = src.replace("phase4A_gradient_diagnostic.csv", "phase4C_probeC_gradient.csv")
src = src.replace("phase4A_effective_rank.csv", "phase4C_probeC_rank.csv")
(ROOT / "scripts/phase4c_probeC_diag.py").write_text(src)
print("generated scripts/phase4c_probeC_diag.py")
PYEOF
"$PY" scripts/phase4c_probeC_diag.py --variant all --batches 8 --bs 2 \
  >> experiments/phase4a/exp4C_diag.log 2>&1
say "diagnostic exit=$?"

say "CHAIN DONE"
