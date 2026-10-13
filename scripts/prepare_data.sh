#!/usr/bin/env bash
# Build the three synthetic pools from one shared set of seed documents.
#
# The first invocation samples and caches data/seed_documents.jsonl; the next
# two reuse it. Do not delete that file between styles, the seed matching is
# the point of the experiment.
#
# Usage: scripts/prepare_data.sh [N_DOCS] [BACKEND]
#   N_DOCS   how many seed documents to rewrite (default 20000)
#   BACKEND  transformers (default) or vllm if you have it installed
set -euo pipefail
cd "$(dirname "$0")/.."

N_DOCS="${1:-20000}"
BACKEND="${2:-transformers}"
GENERATOR="${GENERATOR:-Qwen/Qwen2.5-7B-Instruct}"

echo "generating $N_DOCS documents per style with $GENERATOR via $BACKEND"

for STYLE in rephrase textbook qa; do
  echo
  echo "=== $STYLE ==="
  python -m src.synthesize \
    --style "$STYLE" \
    --n-docs "$N_DOCS" \
    --generator "$GENERATOR" \
    --backend "$BACKEND"
done

echo
echo "synthetic pools:"
wc -l data/synthetic/*.jsonl

cat <<'EOF'

Before you trust these pools, read twenty of them by hand:

  head -n 3 data/synthetic/textbook.jsonl | python -m json.tool

You are checking for refusals, truncation mid-sentence, repeated openings
("Sure, here is...") and hallucinated specifics not in the seed document.
Any of those will show up later as an unexplained bump in your loss curve,
and it is much cheaper to find them now.
EOF
