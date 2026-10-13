"""The four fixed model sizes used across the whole sweep.

Shapes follow the usual small-decoder conventions (Pythia and SmolLM2 style):
width grows with depth, head dimension stays at 64, SwiGLU MLP ratio stays
near 8/3. Nothing here should change between runs. If you edit a shape, every
alpha you already ran becomes incomparable, so bump a new size key instead.
"""

from __future__ import annotations

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

SIZES: dict[str, dict] = {
    "70m": {
        "hidden_size": 512,
        "intermediate_size": 1365,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "approx_non_embed_params": 70_000_000,
    },
    "160m": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "approx_non_embed_params": 160_000_000,
    },
    "410m": {
        "hidden_size": 1024,
        "intermediate_size": 2730,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "approx_non_embed_params": 410_000_000,
    },
    "830m": {
        "hidden_size": 1536,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "approx_non_embed_params": 830_000_000,
    },
}


def build_tokenizer(name: str):
    """Load the shared tokenizer. One tokenizer for every run, no exceptions."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(size: str, vocab_size: int, max_seq_len: int = 1024) -> LlamaForCausalLM:
    """Instantiate a randomly initialized decoder of the requested size."""
    if size not in SIZES:
        raise KeyError(f"unknown size {size!r}; choose from {tuple(SIZES)}")
    shape = {k: v for k, v in SIZES[size].items() if k != "approx_non_embed_params"}

    config = LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        hidden_act="silu",
        **shape,
    )
    return LlamaForCausalLM(config)


def count_params(model) -> dict[str, int]:
    """Report total and non-embedding parameter counts for the log."""
    total = sum(p.numel() for p in model.parameters())
    embed = model.get_input_embeddings().weight.numel()
    # Weights are tied, so the output head is not counted a second time.
    return {"total": total, "non_embedding": total - embed, "embedding": embed}


if __name__ == "__main__":
    tok = build_tokenizer("HuggingFaceTB/SmolLM2-135M")
    for name in SIZES:
        counts = count_params(build_model(name, len(tok)))
        print(
            f"{name:>5}  total {counts['total'] / 1e6:7.1f}M"
            f"  non-embedding {counts['non_embedding'] / 1e6:7.1f}M"
        )
