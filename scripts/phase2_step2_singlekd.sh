#!/usr/bin/env bash
# Phase-2 Step 2: clean paired single-teacher KD recheck on a FIXED 9999 train subset.
# Control (no-KD) and each KD run share the SAME images/protocol (4ep) so the
# KD effect is isolated (fixes Phase-1 confound: KD on 10k vs no-KD on full 69k).
set -u
cd "$(dirname "$0")/.."
PY=/home/mycode/ai_study/gpu_env/bin/python
OUT=experiments/phase2/s2
mkdir -p "$OUT"
st(){ echo "$(date +%H:%M:%S) $*" >> "$OUT/status.txt"; }

# 1) no-KD control on the SAME 9999 images
run() { # tag config
  local tag="$1" cfg="$2"; shift 2
  if [ -f "$OUT/$tag/checkpoint.pt" ]; then st "$tag exists"; return; fi
  st "START $tag"
  $PY training/train.py --config "$cfg" --num-images 9999 --epochs 4 --seed 0 \
      --outdir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  [ -f "$OUT/$tag/checkpoint.pt" ] && st "DONE $tag" || st "FAIL $tag"
}

run s2_nokd       configs/phase2_stage_a_0.235m.yaml
run s2_kd_YOLOP   configs/phase1b_train_stage_a_offkd_yolop.yaml
run s2_kd_TLP     configs/phase1b_train_stage_a_offkd_twinlitenetplus.yaml
run s2_kd_TriLite configs/phase1b_train_stage_a_offkd_trlitenet.yaml
st "P2-STEP2_SINGLE_KD_DONE"
