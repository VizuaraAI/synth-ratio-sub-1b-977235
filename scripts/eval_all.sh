#!/usr/bin/env bash
# Evaluate every run that does not already have an eval.json.
#
# Usage: scripts/eval_all.sh [--downstream] [--novelty]
#   plain            recompute held-out loss with more batches (fast)
#   --downstream     also run the zero-shot lm-eval suite (slow)
#   --novelty        also run the n-gram novelty probe (slow)
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=("$@")

shopt -s nullglob
RUNS=(runs/*/)
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "no runs found under runs/. Run scripts/run_sweep.sh first." >&2
  exit 1
fi

for RUN in "${RUNS[@]}"; do
  RUN="${RUN%/}"
  if [ -f "$RUN/eval.json" ]; then
    echo "skip $RUN (already evaluated; delete eval.json to redo)"
    continue
  fi
  if [ ! -d "$RUN/checkpoint" ]; then
    echo "skip $RUN (no checkpoint)"
    continue
  fi
  echo "=== $RUN ==="
  python -m src.evaluate --run "$RUN" "${EXTRA[@]}"
done

echo
echo "now run: python -m src.analyze --runs runs --out results"
