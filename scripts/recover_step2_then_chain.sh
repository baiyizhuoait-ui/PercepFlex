#!/bin/bash
set -u
cd /home/mycode/ai_study/trac
PY=/home/mycode/ai_study/gpu_env/bin/python
S2=experiments/phase4a/exp4B_danc_z16_e20_s2
CSV=experiments/phase5/phase5_det_e20_results.csv
LOG=experiments/phase5/recover_step2.log
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }
log "RECOVERY START: continue STEP2 (danc seed2) from ep19 ckpt, then full chain STEP3-5"
if awk -F, 'NR>1 && $2=="danc_z16" && $18==2 {f=1} END{exit !f}' "$CSV" 2>/dev/null; then
  log "STEP2 seed2 row already present - skip recovery"
else
  log "STEP2: train +1 epoch from ep19 backup (genuine continuation -> 20ep)"
  "$PY" training/train.py --config configs/phase4b_danc_z16.yaml --outdir "$S2" --epochs 1 --seed 2 --init "$S2/checkpoint.pt.PREKILL.bak" 2>&1 | tail -3 | tee -a "$LOG"
  log "STEP2: patch checkpoint epoch metadata 1 -> 20"
  "$PY" -c "import torch; p='$S2/checkpoint.pt'; ck=torch.load(p, map_location='cpu', weights_only=False); ck['epoch']=20; torch.save(ck, p); print('patched epoch ->', ck['epoch'])" | tee -a "$LOG"
  log "STEP2: runner eval-only (CK_EP==20) -> appends det_e20 row"
  CELLS_OVERRIDE=danc_z16 EPOCHS_OVERRIDE=20 SEED_OVERRIDE=2 CSV_OVERRIDE="$CSV" bash scripts/phase4b_run.sh 2>&1 | tail -8 | tee -a "$LOG"
  log "STEP2 recovery done"
fi
log "Now run full chain (STEP1/2 skipped via resume guards; STEP3 dp2b / STEP4 lane8 / STEP5 da14)"
bash scripts/clean_master_chain.sh 2>&1 | tail -20 | tee -a "$LOG"
log "RECOVERY CHAIN DONE"
