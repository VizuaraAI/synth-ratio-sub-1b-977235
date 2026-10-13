"""One pretraining run for one (model size, synthetic share) pair.

    python -m src.train --config configs/default.yaml \
        --set model.size=160m mixture.alpha=0.05 run.seed=0

Writes runs/<run_name>/ containing config.yaml, metrics.jsonl and a final
checkpoint. Everything downstream (evaluate.py, analyze.py) reads that
directory, so do not rename it by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import load_config, token_budget
from .data import MixedStream, load_pool, resolve_synthetic_pools, tokenize_real_pool, val_batches
from .model import build_model, build_tokenizer, count_params

VAL_BATCHES = 40  # fixed so validation loss is comparable across all runs


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, total_steps: int, cfg: dict) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio`` of the peak."""
    peak = cfg["optim"]["lr"]
    floor = peak * cfg["optim"]["min_lr_ratio"]
    warmup = max(1, int(total_steps * cfg["optim"]["warmup_ratio"]))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def validate(model, pool, seq_len: int, batch_size: int, device) -> float:
    """Mean next-token loss on the held-out REAL validation pool."""
    model.eval()
    losses = []
    for inputs, labels in val_batches(pool, seq_len, batch_size, VAL_BATCHES):
        x = torch.from_numpy(inputs).to(device)
        y = torch.from_numpy(labels).to(device)
        losses.append(model(input_ids=x, labels=y).loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="key.subkey=value")
    parser.add_argument("--max-steps", type=int, default=None, help="override, for smoke tests")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    name = cfg["run"]["name"]
    out_dir = Path(cfg["run"]["out_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    set_seed(cfg["run"]["seed"])
    device = pick_device()
    seq_len = cfg["model"]["max_seq_len"]
    micro_bs = cfg["optim"]["micro_batch_size"]
    accum = cfg["optim"]["grad_accum_steps"]
    tokens_per_step = micro_bs * accum * seq_len

    total_tokens = token_budget(cfg)
    total_steps = args.max_steps or max(1, total_tokens // tokens_per_step)

    # ---------------------------------------------------------------- data
    tokenizer = build_tokenizer(cfg["model"]["tokenizer"])
    cache_dir = cfg["data"]["cache_dir"]
    alpha = cfg["mixture"]["alpha"]

    train_real_path = tokenize_real_pool(
        tokenizer,
        cfg["data"]["real_dataset"],
        cfg["data"]["real_subset"],
        # 1.1x headroom so random sequence starts never run off the end
        target_tokens=int(total_tokens * 1.1),
        cache_dir=cache_dir,
        split_name="train",
    )
    val_real_path = tokenize_real_pool(
        tokenizer,
        cfg["data"]["real_dataset"],
        cfg["data"]["real_subset"],
        target_tokens=cfg["data"]["val_tokens"],
        cache_dir=cache_dir,
        split_name="val",
        skip_docs=2_000_000,  # disjoint from the training documents
    )

    real = load_pool(train_real_path)
    val = load_pool(val_real_path)
    synthetic = (
        resolve_synthetic_pools(
            cfg["mixture"]["synthetic_style"], cache_dir, cfg["data"]["synthetic_dir"], tokenizer
        )
        if alpha > 0.0
        else {}
    )

    stream = MixedStream(
        real=real,
        synthetic=synthetic,
        alpha=alpha,
        seq_len=seq_len,
        batch_size=micro_bs,
        seed=cfg["run"]["seed"],
    )
    passes = stream.epochs_over_synthetic(total_tokens)
    if passes > 4.0:
        print(
            f"[warn] alpha={alpha} consumes the synthetic pool {passes:.1f} times. "
            "Generate more synthetic documents or this run confounds the mixture "
            "effect with memorization of a small pool."
        )

    # --------------------------------------------------------------- model
    model = build_model(cfg["model"]["size"], len(tokenizer), seq_len).to(device)
    counts = count_params(model)
    if cfg["optim"]["compile"]:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    use_bf16 = cfg["optim"]["dtype"] == "bfloat16" and device.type == "cuda"

    run_meta = {
        "run": name,
        "size": cfg["model"]["size"],
        "alpha": alpha,
        "style": cfg["mixture"]["synthetic_style"],
        "seed": cfg["run"]["seed"],
        "params_total": counts["total"],
        "params_non_embedding": counts["non_embedding"],
        "total_tokens": total_tokens,
        "total_steps": total_steps,
        "tokens_per_step": tokens_per_step,
        "synthetic_pool_passes": round(passes, 3),
        "device": str(device),
    }
    print(json.dumps(run_meta, indent=2))

    writer = None
    if cfg["run"]["wandb_project"]:
        import wandb

        writer = wandb.init(project=cfg["run"]["wandb_project"], name=name, config=cfg)

    # -------------------------------------------------------------- train
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w") as metrics:
        metrics.write(json.dumps({"event": "start", **run_meta}) + "\n")
        model.train()
        started = time.time()

        for step in range(total_steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, cfg)

            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(accum):
                inputs, labels = stream.batch()
                x = torch.from_numpy(inputs).to(device)
                y = torch.from_numpy(labels).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    loss = model(input_ids=x, labels=y).loss / accum
                loss.backward()
                accumulated += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            optimizer.step()

            if step % cfg["run"]["log_every"] == 0:
                record = {
                    "event": "train",
                    "step": step,
                    "train_loss": accumulated,
                    "lr": optimizer.param_groups[0]["lr"],
                    "tokens": (step + 1) * tokens_per_step,
                    "elapsed_s": round(time.time() - started, 1),
                }
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(
                    f"step {step:>6}/{total_steps}  loss {accumulated:.4f}"
                    f"  lr {record['lr']:.2e}"
                )
                if writer:
                    writer.log(record, step=step)

            is_last = step == total_steps - 1
            if is_last or (step > 0 and step % cfg["run"]["eval_every"] == 0):
                val_loss = validate(model, val, seq_len, micro_bs, device)
                record = {"event": "val", "step": step, "val_loss": val_loss}
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(f"  -> held-out real validation loss {val_loss:.4f}")
                if writer:
                    writer.log(record, step=step)

        final_val = validate(model, val, seq_len, micro_bs, device)
        summary = {
            "event": "final",
            **run_meta,
            "final_val_loss": final_val,
            "wall_clock_s": round(time.time() - started, 1),
        }
        metrics.write(json.dumps(summary) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if cfg["run"]["save_final_checkpoint"]:
        model.save_pretrained(out_dir / "checkpoint")
        tokenizer.save_pretrained(out_dir / "checkpoint")

    print(f"\nfinal held-out real validation loss: {final_val:.4f}")
    print(f"wrote {out_dir}")
    if writer:
        writer.finish()


if __name__ == "__main__":
    main()
