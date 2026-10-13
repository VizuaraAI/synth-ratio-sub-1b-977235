# synth-ratio-sub-1b

Starter code for **The Optimal Synthetic Data Ratio Below One Billion
Parameters**, by Saideep Koppaka (Vizuara SLM Bootcamp).

## The question

Published work finds that the best fraction of synthetic data in a
pretraining mixture is not a constant but grows with model size: about
**12.5% at 3B** and about **25% at 6B** parameters
([Kang et al., EMNLP 2025](https://arxiv.org/abs/2510.01631)). Extend that
trend downward and it predicts a **near-zero optimum below 1B parameters**,
which is exactly where on-device models live, and exactly where nobody has
measured it.

That prediction is also awkward, because the best-known sub-1B recipes
([phi-1.5](https://arxiv.org/abs/2309.05463),
[TinyStories](https://arxiv.org/abs/2305.07759)) train on corpora that are
almost entirely synthetic. One of those two things is wrong for the general
case, and this repo is how you find out which.

**The experiment.** Pretrain decoders at 70M, 160M, 410M and 830M parameters,
sweeping the synthetic token share `alpha` over
`{0, 2, 5, 10, 15, 25, 50}%` with everything else frozen. Extract
`alpha*(N)`, the loss-minimizing share at each scale. Test whether it rises
with `N`, and whether it is distinguishable from zero at 70M.

**The one design decision that makes this credible:** the synthetic pools are
generated from the *same seed documents* as the real pool. If you compare a
real corpus against a separately sourced synthetic corpus, topic and register
move with `alpha` and you cannot attribute anything. Here only the
presentation changes.

## What is in here

```
configs/default.yaml       one run's settings; alpha and model.size are what you sweep
src/config.py              config loading with dotted-key CLI overrides
src/model.py               the four fixed model sizes (do not edit mid-project)
src/synthesize.py          builds the rephrase / textbook / QA pools from shared seeds
src/data.py                tokenize-once caching and the alpha-weighted mixed stream
src/train.py               one pretraining run for one (size, alpha) pair
src/evaluate.py            held-out real loss, zero-shot tasks, n-gram novelty
src/analyze.py             fits alpha*(N), writes the paper's Table 1 and figure
scripts/smoke_test.sh      30-second plumbing check, run this first
scripts/prepare_data.sh    generate all three synthetic pools
scripts/run_sweep.sh       all 7 alpha values at one model size
scripts/eval_all.sh        evaluate every finished run
experiments/log_template.md  the experiment log, rows map to the paper's tables
```

No data files are committed. The loaders stream public datasets from the
Hugging Face Hub on first use and cache tokenized arrays under `data/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Needed for the gated FineWeb-Edu download:
huggingface-cli login
```

A single GPU with 24GB is enough for the 70M and 160M sweeps, which is where
the paper's claim lives. 410M and 830M want 40GB or more, or gradient
accumulation and patience.

## The first three things to run

**1. Prove the plumbing works (30 seconds).**

```bash
bash scripts/smoke_test.sh
```

This imports every module, prints the four model shapes, and trains 70M for
20 steps on real data only. If this fails, nothing else will work.

**2. Build the synthetic pools (hours, GPU-bound).**

```bash
bash scripts/prepare_data.sh 20000
```

Then **read twenty generations by hand** before you train on them. Refusals,
truncations and repeated openings all show up later as an unexplained bump in
your loss curve, and finding them now is much cheaper.

**3. Run the first real row of the results table.**

```bash
bash scripts/run_sweep.sh 70m
bash scripts/eval_all.sh
python -m src.analyze --runs runs --out results
```

That gives you seven runs, one filled row of Table 1, and your first
`alpha*`. Then run `scripts/run_sweep.sh 70m 1` and `70m 2` to get the seed
noise floor, because until you know it you cannot say whether your `alpha*`
is real.

## Reading the output

`src/analyze.py` writes to `results/`:

- `all_runs.csv` : one row per run
- `alpha_star.json` : `alpha*` per size with a bootstrap 95% interval, plus
  `distinguishable_from_zero` and `beats_real_only` flags
- `table1_main.tex` : LaTeX rows to paste over the placeholders in the paper
- `alpha_star.png` : your version of figure 2

## Things that will bite you

- **Running out of synthetic data.** At `alpha = 0.5` with a small pool you
  will loop over the same generations many times, which confounds the mixture
  effect with memorization. `train.py` warns when the pass count exceeds 4;
  take the warning seriously and generate more.
- **Changing anything mid-sweep.** Learning rate, tokenizer and architecture
  must be identical across every `alpha`. If you change one, every earlier run
  becomes incomparable. Start a new `run.out_dir` instead.
- **Trusting a single seed.** Loss differences between adjacent `alpha` values
  will be small. Without replicates you cannot tell a real minimum from noise.
- **Validating on synthetic text.** The held-out split is real web text only,
  on purpose. Validating on the training distribution would make high `alpha`
  look artificially good.

## A negative result is a result

If no `alpha > 0` beats the real-data-only baseline at 70M, you have not
failed. You have measured the thing the field currently guesses at, and
[Insights from Negative Results in NLP](https://insights-workshop.github.io/)
exists for exactly this. Write it up with the same care either way.
