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

run s2_nokd       configs/phase2_stageA_0.235M.yaml
run s2_kd_YOLOP   configs/train_stageA_offkd_YOLOP.yaml
run s2_kd_TLP     configs/train_stageA_offkd_TwinLiteNetPlus.yaml
run s2_kd_TriLite configs/train_stageA_offkd_TriLiteNet.yaml
st "STEP2_SINGLE_KD_DONE"
