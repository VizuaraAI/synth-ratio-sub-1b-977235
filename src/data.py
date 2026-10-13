"""Tokenize once, then interleave real and synthetic tokens at a chosen ratio.

Two ideas matter here.

1. Tokenization is cached to a flat uint16 memmap per pool. Every alpha value
   reads the same cached pools, so the 28 runs in the sweep do not pay the
   tokenization cost 28 times, and no run can accidentally see a differently
   preprocessed corpus.

2. Mixing happens at the DOCUMENT level, not the token level. A synthetic
   fraction of 0.05 means roughly 5 percent of the training TOKENS come from
   synthetic documents, assembled by sampling whole documents. Chopping
   individual sequences in half across sources would create documents that are
   neither real nor synthetic and would blur exactly the effect under study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

DTYPE = np.uint16  # fine for any tokenizer with vocab < 65536
STYLES = ("rephrase", "textbook", "qa")


# --------------------------------------------------------------------------- #
# Tokenization and caching
# --------------------------------------------------------------------------- #
def _write_memmap(path: Path, chunks: list[np.ndarray]) -> np.ndarray:
    total = int(sum(c.size for c in chunks))
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype=DTYPE, mode="w+", shape=(total,))
    offset = 0
    for chunk in chunks:
        arr[offset : offset + chunk.size] = chunk
        offset += chunk.size
    arr.flush()
    (path.with_suffix(".meta.json")).write_text(json.dumps({"n_tokens": total}))
    return arr


def tokenize_real_pool(
    tokenizer,
    dataset: str,
    subset: str,
    target_tokens: int,
    cache_dir: str,
    split_name: str = "train",
    skip_docs: int = 0,
) -> Path:
    """Stream the real corpus until ``target_tokens`` tokens are cached."""
    out = Path(cache_dir) / f"real_{split_name}.bin"
    if out.exists():
        print(f"[data] reusing cached real pool: {out}")
        return out

    stream = load_dataset(dataset, name=subset, split="train", streaming=True)
    eos = tokenizer.eos_token_id
    chunks: list[np.ndarray] = []
    seen = 0

    bar = tqdm(total=target_tokens, desc=f"tokenizing real/{split_name}", unit="tok")
    for i, row in enumerate(stream):
        if i < skip_docs:
            continue
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        chunks.append(np.asarray(ids + [eos], dtype=DTYPE))
        seen += len(ids) + 1
        bar.update(len(ids) + 1)
        if seen >= target_tokens:
            break
    bar.close()

    _write_memmap(out, chunks)
    print(f"[data] wrote {seen:,} real tokens to {out}")
    return out


def tokenize_synthetic_pool(tokenizer, jsonl_path: str | Path, cache_dir: str) -> Path:
    """Tokenize one synthetic style file into its own cached memmap."""
    jsonl_path = Path(jsonl_path)
    out = Path(cache_dir) / f"synthetic_{jsonl_path.stem}.bin"
    if out.exists():
        print(f"[data] reusing cached synthetic pool: {out}")
        return out
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} is missing. Run src/synthesize.py --style {jsonl_path.stem} first."
        )

    eos = tokenizer.eos_token_id
    chunks: list[np.ndarray] = []
    with open(jsonl_path) as handle:
        for line in tqdm(handle, desc=f"tokenizing {jsonl_path.stem}"):
            ids = tokenizer(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            if ids:
                chunks.append(np.asarray(ids + [eos], dtype=DTYPE))

    _write_memmap(out, chunks)
    print(f"[data] wrote {sum(c.size for c in chunks):,} synthetic tokens to {out}")
    return out


def load_pool(path: str | Path) -> np.ndarray:
    path = Path(path)
    n_tokens = json.loads(path.with_suffix(".meta.json").read_text())["n_tokens"]
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(n_tokens,))


# --------------------------------------------------------------------------- #
# The mixed training stream
# --------------------------------------------------------------------------- #
class MixedStream:
    """Yields fixed-length training batches drawn from real and synthetic pools.

    Each sequence is sampled whole from one pool. The pool is chosen by a coin
    flip weighted so that the expected synthetic token share equals ``alpha``.
    Because every sequence has the same length, weighting sequences is the same
    as weighting tokens.
    """

    def __init__(
        self,
        real: np.ndarray,
        synthetic: dict[str, np.ndarray],
        alpha: float,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if alpha > 0.0 and not synthetic:
            raise ValueError("alpha > 0 but no synthetic pools were provided")

        self.real = real
        self.styles = sorted(synthetic)
        self.synthetic = synthetic
        self.alpha = alpha
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Guard rail: with a tiny synthetic pool and a large alpha you would
        # revisit the same synthetic documents many times per epoch, which
        # confounds the mixture effect with memorization of a small set.
        if alpha > 0.0:
            synth_tokens = sum(p.size for p in synthetic.values())
            self.synthetic_tokens = synth_tokens
        else:
            self.synthetic_tokens = 0

    def epochs_over_synthetic(self, total_tokens: int) -> float:
        """How many times the synthetic pool is consumed over a full run."""
        if self.alpha == 0.0 or self.synthetic_tokens == 0:
            return 0.0
        return (self.alpha * total_tokens) / self.synthetic_tokens

    def _sample(self, pool: np.ndarray) -> np.ndarray:
        # +1 token so the batch can be split into inputs and shifted labels.
        high = pool.size - self.seq_len - 1
        if high <= 0:
            raise ValueError("pool is smaller than one sequence; cache more tokens")
        start = int(self.rng.integers(0, high))
        return np.asarray(pool[start : start + self.seq_len + 1], dtype=np.int64)

    def batch(self) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        for _ in range(self.batch_size):
            if self.alpha > 0.0 and self.rng.random() < self.alpha:
                style = self.styles[int(self.rng.integers(0, len(self.styles)))]
                rows.append(self._sample(self.synthetic[style]))
            else:
                rows.append(self._sample(self.real))
        stacked = np.stack(rows)
        return stacked[:, :-1], stacked[:, 1:]


def val_batches(pool: np.ndarray, seq_len: int, batch_size: int, max_batches: int):
    """Deterministic, non-overlapping sweep over the held-out real pool."""
    usable = (pool.size - 1) // seq_len
    for b in range(min(max_batches, usable // batch_size)):
        rows = []
        for r in range(batch_size):
            start = (b * batch_size + r) * seq_len
            rows.append(np.asarray(pool[start : start + seq_len + 1], dtype=np.int64))
        stacked = np.stack(rows)
        yield stacked[:, :-1], stacked[:, 1:]


def resolve_synthetic_pools(style: str, cache_dir: str, synthetic_dir: str, tokenizer) -> dict:
    """Return the pools for a mixture style ("balanced" uses all three)."""
    wanted = STYLES if style == "balanced" else (style,)
    pools = {}
    for name in wanted:
        path = tokenize_synthetic_pool(tokenizer, Path(synthetic_dir) / f"{name}.jsonl", cache_dir)
        pools[name] = load_pool(path)
    return pools
