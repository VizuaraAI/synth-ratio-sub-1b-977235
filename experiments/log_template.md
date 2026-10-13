# Experiment log

One row per pretraining run. The rows map directly onto the paper's tables,
so a complete log is a filled-in results section. Copy the row format exactly
and do not delete failed runs: a run you killed at step 3000 because the loss
diverged is evidence about the learning rate, and it belongs here.

Fill this in the same day you run the experiment. Reconstructing it a week
later from `runs/` is possible but you will have lost the reason you ran it.

---

## Table 1: main sweep (balanced synthetic style)

Maps to **Table 1** of the paper. `val_loss` is held-out real-web loss from
`eval.json`. Leave `alpha*` and `delta_best` blank until the row is complete,
then paste the numbers that `python -m src.analyze` prints.

| run id | size | alpha | style | seed | tokens | val_loss | ppl | wall clock | GPU | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 70m_balanced_alpha0_seed0 | 70M | 0.00 | balanced | 0 | 1.4B | | | | | real-data-only baseline |
| 70m_balanced_alpha2_seed0 | 70M | 0.02 | balanced | 0 | 1.4B | | | | | |
| 70m_balanced_alpha5_seed0 | 70M | 0.05 | balanced | 0 | 1.4B | | | | | |
| 70m_balanced_alpha10_seed0 | 70M | 0.10 | balanced | 0 | 1.4B | | | | | |
| 70m_balanced_alpha15_seed0 | 70M | 0.15 | balanced | 0 | 1.4B | | | | | |
| 70m_balanced_alpha25_seed0 | 70M | 0.25 | balanced | 0 | 1.4B | | | | | the published 6B optimum |
| 70m_balanced_alpha50_seed0 | 70M | 0.50 | balanced | 0 | 1.4B | | | | | |
| 160m_balanced_alpha0_seed0 | 160M | 0.00 | balanced | 0 | 3.2B | | | | | |
| ... | | | | | | | | | | |

**Row summary** (one line per model size, fills the last two columns of Table 1):

| size | alpha* | 95% CI | loss at alpha* | loss at alpha=0 | delta_best | beats real-only? |
|---|---|---|---|---|---|---|
| 70M | | | | | | |
| 160M | | | | | | |
| 410M | | | | | | |
| 830M | | | | | | |

---

## Table 2: downstream accuracy

Maps to **Table 2**. Run `scripts/eval_all.sh --downstream` to populate.

| size | alpha | ARC-E | HellaSwag | PIQA | SciQ | LAMBADA | mean |
|---|---|---|---|---|---|---|---|
| 70M | 0.00 | | | | | | |
| 70M | alpha* | | | | | | |
| 70M | 0.25 | | | | | | |

---

## Table 3: style ablation at 160M

Maps to **Table 3**. Three extra sweeps with `mixture.synthetic_style` set to
each single style.

| style | alpha=0.05 delta | alpha=0.25 delta | 8-gram novelty at 0.25 | notes |
|---|---|---|---|---|
| rephrase | | | | |
| textbook | | | | |
| qa | | | | |
| balanced | | | | |

---

## Seed noise floor

You cannot claim an optimum is nonzero until you know how much two identical
runs differ. Fill this before you interpret any curve.

| size | alpha | seed 0 | seed 1 | seed 2 | std dev |
|---|---|---|---|---|---|
| 70M | 0.00 | | | | |
| 70M | 0.05 | | | | |
| 160M | 0.00 | | | | |
| 160M | 0.05 | | | | |

**Noise floor used in the paper:** _____ nats. Any `delta_best` smaller than
this is reported as "not distinguishable from the real-data-only baseline".

---

## Running decisions log

Dated notes on anything that changed, and why. This becomes the methodology
section and saves you from "why is run 14 different?" three weeks from now.

### YYYY-MM-DD
- **Decision:**
- **Why:**
- **Affects which runs:**
- **Does this invalidate earlier runs?** (if yes, list them and re-run)

---

## Surprises and dead ends

Things that did not work. This is the most valuable page in the log and the
raw material for the paper's limitations section.

- **What I expected:**
- **What happened:**
- **What I checked to rule out a bug:**
- **What I concluded:**
