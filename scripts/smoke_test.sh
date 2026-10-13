#!/usr/bin/env bash
# Thirty-second sanity check. Trains the smallest model for 20 steps at two
# alpha values on a tiny slice of data. It proves the plumbing works; it
# proves nothing about the science. Run this before anything else, and run it
# again any time you touch src/data.py.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== importing every module =="
python -c "import src.config, src.data, src.model, src.train, src.evaluate, src.analyze; print('ok')"

echo "== model shapes =="
python -m src.model

echo "== 20 steps at alpha = 0 (real data only, no synthetic pool needed) =="
python -m src.train \
  --set model.size=70m mixture.alpha=0.0 run.out_dir=runs_smoke \
        optim.micro_batch_size=2 optim.grad_accum_steps=1 data.val_tokens=200000 \
  --max-steps 20

if [ -f data/synthetic/rephrase.jsonl ]; then
  echo "== 20 steps at alpha = 0.25 (mixed stream) =="
  python -m src.train \
    --set model.size=70m mixture.alpha=0.25 mixture.synthetic_style=rephrase \
          run.out_dir=runs_smoke optim.micro_batch_size=2 optim.grad_accum_steps=1 \
          data.val_tokens=200000 \
    --max-steps 20
else
  echo "== skipping the mixed-stream check: no data/synthetic/rephrase.jsonl yet =="
  echo "   run scripts/prepare_data.sh first"
fi

echo
echo "smoke test passed. Delete runs_smoke/ when you are done."
