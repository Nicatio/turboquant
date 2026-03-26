from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from turboquant.hf_cache import resolve_cached_model_path


def build_prompt(prompt: str, use_chat_template: bool) -> str:
    if use_chat_template:
        return prompt
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a local MLX model and run a short generation.")
    parser.add_argument("--model", required=True, help="HF/MLX model repo, e.g. mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--prompt", default="Explain what vector quantization is in one short paragraph.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    started = time.time()
    model, tokenizer = load(
        resolved_model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )
    loaded_at = time.time()

    prompt = build_prompt(args.prompt, use_chat_template=False)
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temp)
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=False,
    )
    finished_at = time.time()

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"load_seconds={loaded_at - started:.2f}")
    print(f"generation_seconds={finished_at - loaded_at:.2f}")
    print("--- output ---")
    print(text)


if __name__ == "__main__":
    main()
