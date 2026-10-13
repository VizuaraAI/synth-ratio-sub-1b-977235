# CLAUDE.md

Instructions for the coding agent working in this repository with Saideep.

## What this project is

A controlled measurement of **`alpha*(N)`**: the fraction of synthetic data
in a pretraining mixture that minimizes held-out real-web validation loss, as
a function of model size `N`, for `N` below one billion parameters.

The motivating result is that `alpha*` is reported at roughly 12.5% for 3B
and 25% for 6B parameter models (Kang et al., EMNLP 2025,
arXiv:2510.01631). Extrapolated downward, that trend predicts `alpha*`
approaches zero below 1B. Nobody has measured it there. This repo measures
it.

The deliverable is a paper. Every change should make some row of
`experiments/log_template.md` easier to fill in.

## The invariants

These are not style preferences. Breaking one invalidates the experiment, so
if a request would break one, **say so before doing it**.

1. **Only `mixture.alpha` and `model.size` vary across runs.** Learning rate,
   schedule, batch size in tokens, tokenizer, architecture family,
   initialization and the tokens-per-parameter rule are frozen. If a run
   needs different settings, it is a different experiment and needs a
   different `run.out_dir`.
2. **Synthetic pools are seed-matched to the real pool.** All three styles
   rewrite the same `data/seed_documents.jsonl`, and the originals stay in
   the real pool. Never regenerate seeds for one style only.
3. **Validation is real web text only.** `data/real_val.bin` must never
   contain synthetic tokens, and must come from documents disjoint from the
   training split.
4. **Token budget scales with parameters, not held constant.** 20 tokens per
   non-embedding parameter at every size. This is what makes `alpha*`
   comparable across sizes.
5. **Mixing is at the document level.** A training sequence comes entirely
   from the real pool or entirely from one synthetic pool. Do not splice.

## Layout

| path | role |
|---|---|
| `configs/default.yaml` | the only place run settings live |
| `src/config.py` | loading plus `key.subkey=value` CLI overrides, validation |
| `src/model.py` | `SIZES` dict, the four frozen architectures |
| `src/synthesize.py` | seed sampling and the three generation prompts |
| `src/data.py` | tokenize-once memmap cache, `MixedStream`, validation batches |
| `src/train.py` | one run, writes `runs/<name>/{config.yaml,metrics.jsonl,summary.json,checkpoint/}` |
| `src/evaluate.py` | writes `runs/<name>/eval.json` |
| `src/analyze.py` | reads all runs, writes `results/` including `table1_main.tex` |
| `experiments/log_template.md` | the experiment log, rows map to paper tables |

Run naming is `{size}_{style}_alpha{pct}_seed{n}` and is generated in
`src/config.py:run_name`. Downstream tooling parses these directories, so do
not rename them by hand.

## Conventions

- Python 3.10+, type hints on function signatures, `from __future__ import
  annotations` at the top of each module.
- Standard library, then third party, then local imports.
- Every script is runnable as `python -m src.<module>` with `argparse` and a
  module docstring showing a real invocation.
- Config values are read from the config dict, never hardcoded in `train.py`.
- Comments explain *why*, especially where a choice protects an invariant.
  Do not add comments that restate the code.
- No new dependencies without asking. The point of this repo is that it reads
  in one sitting.

## Working with Saideep

- He is doing this for the first time over roughly eight weeks of self-paced
  work alongside other commitments. Explain the reasoning behind a change, not
  just the diff.
- Prefer the smallest change that answers the question. A 70M run that
  finishes tonight beats an 830M run that finishes next week.
- When a result looks surprising, help him rule out a bug before he gets
  excited: check the pass count over the synthetic pool, check the validation
  split is real-only, check the seed noise floor, check that the loss curve
  is not still descending at the end of training.
- If he asks for something that would break an invariant above, flag it
  first, explain which run comparisons it would invalidate, then do it if he
  still wants it.
- Do not inflate results. If `delta_best` is inside the seed noise band, the
  honest statement is "not distinguishable from the real-data-only baseline",
  and that sentence belongs in the paper.

## Current state

The scaffold runs end to end but no real experiment has been run yet.

## Next steps, in order

1. `bash scripts/smoke_test.sh` and fix anything it surfaces.
2. `bash scripts/prepare_data.sh 20000`, then hand-read 20 generations per
   style and note what you see in the experiment log.
3. `bash scripts/run_sweep.sh 70m` : the full 7-point alpha curve at the
   smallest size. This is the single most informative thing in the project.
4. Seed replicates: `scripts/run_sweep.sh 70m 1` and `70m 2`. Establish the
   noise floor before interpreting any curve.
5. `bash scripts/run_sweep.sh 160m`, then the 410M and 830M rows as compute
   allows.
6. Style ablation at 160M: `scripts/run_sweep.sh 160m 0 rephrase`, then
   `textbook`, then `qa`.
7. `python -m src.analyze` and paste `results/table1_main.tex` into the paper.

## Useful commands

```bash
bash scripts/smoke_test.sh                        # plumbing check
python -m src.model                               # print the four model shapes
python -m src.train --set model.size=70m mixture.alpha=0.05 --max-steps 50
python -m src.evaluate --run runs/70m_balanced_alpha5_seed0 --downstream
python -m src.analyze --runs runs --out results
```
