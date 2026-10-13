"""Build the three synthetic pools from the SAME seed documents as the real pool.

This is the methodological point of the whole project. If your synthetic data
came from different source documents than your real data, then a loss
difference between alpha = 0 and alpha = 0.25 could be a topic shift rather
than a real-versus-synthetic effect. So: sample seed documents once, keep the
originals for the real pool, and rewrite those same documents in three styles.

Styles
    rephrase   paraphrase the document, same facts, cleaner prose (WRAP-like)
    textbook   expository rewrite with definitions and a worked explanation
    qa         a handful of question/answer pairs grounded in the document

Run with a local generator through vLLM when you have the GPU for it, and with
the slower transformers path when you do not. Both write the same JSONL shape:

    {"id": ..., "style": ..., "seed_id": ..., "text": ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

PROMPTS: dict[str, str] = {
    "rephrase": (
        "Rewrite the passage below in clear, well-organized English. Keep every "
        "fact that is present and do not add facts that are not. Do not comment "
        "on the task, output only the rewritten passage.\n\nPassage:\n{doc}\n\n"
        "Rewritten passage:"
    ),
    "textbook": (
        "Turn the passage below into a short textbook section for a motivated "
        "undergraduate. Define the key terms, explain the central idea step by "
        "step, and end with one worked example. Stay grounded in the passage and "
        "do not invent specifics. Output only the section.\n\nPassage:\n{doc}\n\n"
        "Textbook section:"
    ),
    "qa": (
        "Write three question and answer pairs that can be answered using only "
        "the passage below. Each answer should be two to four sentences. Format "
        "each pair as 'Q: ...' on one line and 'A: ...' on the next. Output only "
        "the pairs.\n\nPassage:\n{doc}\n\nPairs:"
    ),
}

# Seed docs shorter than this are too thin to rewrite usefully; longer ones get
# truncated so the generator prompt stays inside a sane context window.
MIN_SEED_CHARS = 1_000
MAX_SEED_CHARS = 6_000


def load_seed_documents(
    dataset: str, subset: str, n_docs: int, cache_dir: str, seed: int = 0
) -> list[dict]:
    """Stream the real corpus and keep the first ``n_docs`` usable documents."""
    stream = load_dataset(dataset, name=subset, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=10_000)

    docs: list[dict] = []
    for i, row in enumerate(stream):
        text = row.get("text", "")
        if len(text) < MIN_SEED_CHARS:
            continue
        docs.append({"seed_id": f"seed-{len(docs):08d}", "text": text[:MAX_SEED_CHARS]})
        if len(docs) >= n_docs:
            break

    seed_path = Path(cache_dir) / "seed_documents.jsonl"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seed_path, "w") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    print(f"wrote {len(docs)} seed documents to {seed_path}")
    return docs


def generate_vllm(prompts: list[str], model: str, max_new_tokens: int) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.90)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    return [out.outputs[0].text.strip() for out in llm.generate(prompts, params)]


def generate_transformers(
    prompts: list[str], model: str, max_new_tokens: int, batch_size: int = 4
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    net.eval()

    outputs: list[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generating"):
        batch = prompts[start : start + batch_size]
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            ids = net.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = ids[:, enc["input_ids"].shape[1] :]
        outputs.extend(t.strip() for t in tokenizer.batch_decode(new_ids, skip_special_tokens=True))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--n-docs", type=int, default=20_000)
    parser.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="transformers")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--real-dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--real-subset", default="sample-10BT")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_path = Path(args.cache_dir) / "seed_documents.jsonl"
    if seed_path.exists():
        with open(seed_path) as handle:
            docs = [json.loads(line) for line in handle][: args.n_docs]
        print(f"reusing {len(docs)} cached seed documents (same seeds for every style)")
    else:
        docs = load_seed_documents(
            args.real_dataset, args.real_subset, args.n_docs, args.cache_dir, args.seed
        )

    prompts = [PROMPTS[args.style].format(doc=d["text"]) for d in docs]
    runner = generate_vllm if args.backend == "vllm" else generate_transformers
    texts = runner(prompts, args.generator, args.max_new_tokens)

    out_path = Path(args.out_dir) / f"{args.style}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out_path, "w") as handle:
        for i, (doc, text) in enumerate(zip(docs, texts)):
            if len(text) < 200:  # a refusal or an empty generation
                continue
            handle.write(
                json.dumps(
                    {
                        "id": f"{args.style}-{i:08d}",
                        "style": args.style,
                        "seed_id": doc["seed_id"],
                        "text": text,
                    }
                )
                + "\n"
            )
            kept += 1
    print(f"wrote {kept}/{len(texts)} generations to {out_path}")


if __name__ == "__main__":
    main()
