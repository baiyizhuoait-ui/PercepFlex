#!/bin/bash
# Phase 6: combo 3-seed confirmation (seed0 done in phase6_results.csv)
set -u
cd /home/mycode/ai_study/trac
CSV=experiments/phase6/phase6_results.csv
[ -f experiments/phase4a/STOP_CHAIN ] && { echo "[seeds] STOP seen"; exit 0; }
for S in 1 2; do
  if awk -F, -v s="$S" "NR>1 && \$2==\"combo20_z16\" && \$18==s {f=1} END{exit !f}" "$CSV"; then
    echo "[seeds] combo20 seed$s already done"; continue
  fi
  bash scripts/phase6_run.sh combo20_s$s configs/phase6_combo_danc_l14f1.yaml 20 $S "$CSV" r2 combo20_z16 16 || true
  [ -f experiments/phase4a/STOP_CHAIN ] && { echo "[seeds] STOP seen"; exit 0; }
done
echo "[seeds] 3-SEED COMPLETE at $(date "+%F %T")"
