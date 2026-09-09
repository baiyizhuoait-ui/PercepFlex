#!/bin/bash
# Phase 6 overnight GPU-2 chain (2026-09-10 01:30 -> ~08:00)
# EXP-4: combo (danc anchors + l14f1 lane 1/4) 4ep screen -> 20ep confirm (gated)
# EXP-1: lch72 (FLOPs-parity channel control) 4ep -> 20ep only if it beats l14f1 4ep
# Gates are preregistered falsification bars; STOP file supported.
set -u
cd /home/mycode/ai_study/trac
P6=experiments/phase6
CSV=$P6/phase6_results.csv
STOP=experiments/phase4a/STOP_CHAIN
mkdir -p "$P6"
[ -f "$CSV" ] || echo "variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit" > "$CSV"

maybe_stop() { [ -f "$STOP" ] && { echo "[chain] STOP_CHAIN seen - clean exit at $(date '+%F %T')"; exit 0; }; }

# reference values (from phase5 CSVs, committed):
#   l14f1 4ep: lane_fg 0.1916  lane_mIoU 0.5839   (d04f7d4)
#   l14f1 20ep: lane_fg 0.2192  lane_mIoU 0.5988  (f775718)
get_val() { awk -F, -v c="$1" -v col="$2" 'NR==1{for(i=1;i<=NF;i++) if($i==col) ic=i} $2==c && ic {v=$ic} END{print v}' "$CSV"; }

echo "[chain] STEP A: combo 4ep screen"
maybe_stop
bash scripts/phase6_run.sh combo4 configs/phase6_combo_danc_l14f1.yaml 4 0 "$CSV" r2 combo4_z16 16 || true
maybe_stop

echo "[chain] STEP B: lch72 4ep (EXP-1 strict parity)"
maybe_stop
bash scripts/phase6_run.sh lch72 configs/phase6_lch72_z16.yaml 4 0 "$CSV" r2 lch72_z16 16 || true
maybe_stop

# GATE 1 (preregistered): combo 4ep must not collapse. det>=0.30 (4ep danc ref
# 0.3954; -25% headroom) AND lane>=0.55 (r2u 4ep 0.5761). Else skip 20ep.
COMBO_DET=$(get_val combo4_z16 mAP50); COMBO_LANE=$(get_val combo4_z16 lane_mIoU)
RUN_COMBO20=1
if awk -v d="$COMBO_DET" 'BEGIN{exit !(d<0.30)}' 2>/dev/null || [ -z "$COMBO_DET" ] || [ "$COMBO_DET" = "NA" ]; then RUN_COMBO20=0; fi
if [ -n "$COMBO_LANE" ] && [ "$COMBO_LANE" != "NA" ] && awk -v l="$COMBO_LANE" 'BEGIN{exit !(l<0.55)}'; then RUN_COMBO20=0; fi
echo "[chain] GATE1 combo4: det=$COMBO_DET lane=$COMBO_LANE -> combo20ep=$RUN_COMBO20"

if [ "$RUN_COMBO20" = "1" ]; then
  echo "[chain] STEP C: combo 20ep confirmation"
  maybe_stop
  bash scripts/phase6_run.sh combo20 configs/phase6_combo_danc_l14f1.yaml 20 0 "$CSV" r2 combo20_z16 16 || true
  maybe_stop
fi

# GATE 2: lch72 20ep only if its 4ep lane_fg beats l14f1 4ep (0.1916).
LCH_FG=$(get_val lch72_z16 lane_fg)
RUN_LCH20=0
if [ -n "$LCH_FG" ] && [ "$LCH_FG" != "NA" ] && awk -v v="$LCH_FG" 'BEGIN{exit !(v>0.1916)}'; then RUN_LCH20=1; fi
echo "[chain] GATE2 lch72 4ep lane_fg=$LCH_FG (l14f1 ref 0.1916) -> lch20ep=$RUN_LCH20"

if [ "$RUN_LCH20" = "1" ]; then
  echo "[chain] STEP D: lch72 20ep (channel arm wins 4ep - needs confirm)"
  maybe_stop
  bash scripts/phase6_run.sh lch72_20 configs/phase6_lch72_z16.yaml 20 0 "$CSV" r2 lch72_20_z16 16 || true
  maybe_stop
fi

echo "[chain] PHASE 6 OVERNIGHT CHAIN COMPLETE at $(date '+%F %T')"
