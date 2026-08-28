#!/usr/bin/env bash
# Run Experiment 1 (Static vs Dynamic) once Stage D training completes.
set -u
cd "$(dirname "$0")/.."
./scripts/exp1_static_vs_dynamic.sh experiments/exp_train_D/checkpoint.pt
