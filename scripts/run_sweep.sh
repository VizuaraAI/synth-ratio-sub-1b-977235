#!/usr/bin/env bash
# The main experiment: every alpha at one model size, run in sequence.
#
# Usage: scripts/run_sweep.sh SIZE [SEED] [STYLE]
#   scripts/run_sweep.sh 70m          # the 7 runs that fill row 1 of Table 1
#   scripts/run_sweep.sh 70m 1        # seed replicate for the noise floor
#   scripts/run_sweep.sh 160m 0 rephrase   # the style ablation
#
# Run the smallest size first and all the way through. A complete 70M row
# teaches you more than four half-finished rows, and it is the row that
# carries the paper's claim.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${1:?usage: scripts/run_sweep.sh SIZE [SEED] [STYLE]}"
SEED="${2:-0}"
STYLE="${3:-balanced}"
ALPHAS=(0.0 0.02 0.05 0.10 0.15 0.25 0.50)

echo "sweeping size=$SIZE seed=$SEED style=$STYLE over ${#ALPHAS[@]} alpha values"
START=$(date +%s)

for ALPHA in "${ALPHAS[@]}"; do
  echo
  echo "=============================================================="
  echo " size=$SIZE  alpha=$ALPHA  style=$STYLE  seed=$SEED"
  echo "=============================================================="
  python -m src.train --set \
    "model.size=$SIZE" \
    "mixture.alpha=$ALPHA" \
    "mixture.synthetic_style=$STYLE" \
    "run.seed=$SEED"
done

echo
echo "sweep finished in $(( ($(date +%s) - START) / 60 )) minutes"
echo "next: scripts/eval_all.sh, then python -m src.analyze --runs runs --out results"
