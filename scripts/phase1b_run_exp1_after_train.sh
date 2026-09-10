#!/usr/bin/env bash
# Run Experiment 1 (Static vs Dynamic) once Stage D training completes.
set -u
cd "$(dirname "$0")/.."
./scripts/phase1_exp1_static_vs_dynamic.sh experiments/phase1b/exp_train_d/checkpoint.pt
