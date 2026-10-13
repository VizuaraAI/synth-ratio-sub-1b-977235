"""Config loading with dotted-key overrides from the command line."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

VALID_STYLES = ("rephrase", "textbook", "qa", "balanced")


def _coerce(text: str) -> Any:
    """Turn a command-line string into an int, float, bool, None or str."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"unknown config section: {part!r} in {dotted_key!r}")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key: {dotted_key!r}")
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load a run config, apply ``key.subkey=value`` overrides, then validate.

    >>> cfg = load_config(overrides=["mixture.alpha=0.25"])  # doctest: +SKIP
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as handle:
        cfg = yaml.safe_load(handle)

    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.subkey=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(raw.strip()))

    validate_config(cfg)
    if cfg["run"]["name"] is None:
        cfg["run"]["name"] = run_name(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    alpha = cfg["mixture"]["alpha"]
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"mixture.alpha must be in [0, 1], got {alpha}")

    style = cfg["mixture"]["synthetic_style"]
    if style not in VALID_STYLES:
        raise ValueError(f"mixture.synthetic_style must be one of {VALID_STYLES}, got {style!r}")

    from .model import SIZES  # local import keeps torch out of config-only use

    size = cfg["model"]["size"]
    if size not in SIZES:
        raise ValueError(f"model.size must be one of {tuple(SIZES)}, got {size!r}")

    if cfg["optim"]["micro_batch_size"] < 1 or cfg["optim"]["grad_accum_steps"] < 1:
        raise ValueError("batch size and grad accumulation steps must both be >= 1")


def run_name(cfg: dict) -> str:
    """Stable, filesystem-safe identifier for a single run."""
    alpha_pct = round(cfg["mixture"]["alpha"] * 100, 1)
    return (
        f"{cfg['model']['size']}"
        f"_{cfg['mixture']['synthetic_style']}"
        f"_alpha{alpha_pct:g}"
        f"_seed{cfg['run']['seed']}"
    )


def token_budget(cfg: dict) -> int:
    """Training tokens for this run: a fixed multiple of non-embedding params."""
    from .model import SIZES

    params = SIZES[cfg["model"]["size"]]["approx_non_embed_params"]
    return int(params * cfg["data"]["tokens_per_param"])
