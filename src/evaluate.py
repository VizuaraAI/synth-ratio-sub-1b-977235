"""Evaluate a finished run: held-out real loss, zero-shot tasks, diversity.

    python -m src.evaluate --run runs/160m_balanced_alpha5_seed0
    python -m src.evaluate --run runs/160m_balanced_alpha5_seed0 --downstream

The loss number is already written by train.py; this script recomputes it
with more validation batches and adds the secondary and diagnostic metrics
that fill Table 2 and the ablation table of the paper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_pool, val_batches

DOWNSTREAM_TASKS = ["arc_easy", "hellaswag", "piqa", "sciq", "lambada_openai"]


def held_out_loss(model, tokenizer, cfg: dict, device, n_batches: int = 200) -> dict:
    pool = load_pool(Path(cfg["data"]["cache_dir"]) / "real_val.bin")
    seq_len = cfg["model"]["max_seq_len"]
    batch_size = cfg["optim"]["micro_batch_size"]

    losses = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_batches(pool, seq_len, batch_size, n_batches):
            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(labels).to(device)
            losses.append(model(input_ids=x, labels=y).loss.item())

    mean = float(np.mean(losses))
    # Standard error over batches, so you can tell a real gap from noise.
    stderr = float(np.std(losses) / np.sqrt(len(losses)))
    return {
        "val_loss": mean,
        "val_loss_stderr": stderr,
        "val_perplexity": float(np.exp(mean)),
        "n_batches": len(losses),
    }


def ngram_novelty(model, tokenizer, cfg: dict, device, n: int = 8, n_samples: int = 200) -> dict:
    """Fraction of generated n-grams that are not in the real training pool.

    A mixture that lowers loss while lowering novelty is buying fluency with
    coverage, which is the failure mode worth catching at small scale.
    """
    pool = load_pool(Path(cfg["data"]["cache_dir"]) / "real_train.bin")
    # Reference set from a bounded slice; the full pool will not fit in memory.
    reference_tokens = np.asarray(pool[: 20_000_000], dtype=np.int64)
    reference = set()
    for i in range(0, len(reference_tokens) - n, n // 2):
        reference.add(tuple(reference_tokens[i : i + n].tolist()))

    prompt = torch.tensor([[tokenizer.bos_token_id or tokenizer.eos_token_id]], device=device)
    novel, total, distinct = 0, 0, Counter()
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            out = model.generate(
                prompt,
                max_new_tokens=128,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            ids = out[0].tolist()
            for i in range(len(ids) - n):
                gram = tuple(ids[i : i + n])
                distinct[gram] += 1
                total += 1
                if gram not in reference:
                    novel += 1

    return {
        f"novelty_{n}gram": novel / total if total else float("nan"),
        f"distinct_{n}gram_ratio": len(distinct) / total if total else float("nan"),
        "n_samples": n_samples,
    }


def run_downstream(checkpoint: Path, batch_size: int = 16) -> dict:
    """Shell out to lm-eval for the zero-shot suite and parse its JSON."""
    out_dir = checkpoint.parent / "lm_eval"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint}",
        "--tasks", ",".join(DOWNSTREAM_TASKS),
        "--num_fewshot", "0",
        "--batch_size", str(batch_size),
        "--output_path", str(out_dir),
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results_files = sorted(out_dir.rglob("results*.json"))
    if not results_files:
        raise FileNotFoundError(f"lm-eval wrote no results under {out_dir}")
    payload = json.loads(results_files[-1].read_text())["results"]

    scores = {}
    for task, metrics in payload.items():
        for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
            if key in metrics:
                scores[task] = float(metrics[key])
                break
    scores["mean"] = float(np.mean(list(scores.values()))) if scores else float("nan")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="a runs/<name> directory")
    parser.add_argument("--downstream", action="store_true", help="also run lm-eval tasks")
    parser.add_argument("--novelty", action="store_true", help="also run n-gram novelty probe")
    parser.add_argument("--val-batches", type=int, default=200)
    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    checkpoint = run_dir / "checkpoint"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint in {run_dir}; train with save_final_checkpoint=true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    results = {
        "run": run_dir.name,
        "size": cfg["model"]["size"],
        "alpha": cfg["mixture"]["alpha"],
        "style": cfg["mixture"]["synthetic_style"],
        "seed": cfg["run"]["seed"],
    }
    results.update(held_out_loss(model, tokenizer, cfg, device, args.val_batches))
    if args.novelty:
        results.update(ngram_novelty(model, tokenizer, cfg, device))
    if args.downstream:
        results["downstream"] = run_downstream(checkpoint, cfg["optim"]["micro_batch_size"])

    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
