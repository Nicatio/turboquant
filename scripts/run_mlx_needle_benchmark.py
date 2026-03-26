from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Dict, List

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from turboquant.benchmark_utils import (
    bytes_to_gb,
    cache_nbytes,
    contains_answer,
    encode_chat_prompt,
    exact_match,
)
from turboquant.hf_cache import resolve_cached_model_path
from turboquant.kv_cache import TurboQuantDirectKVCache, TurboQuantKVCache
from turboquant.mlx_attention import enable_turboquant_direct_attention, get_transformer_layers


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SYSTEM_PROMPT = (
    "You answer retrieval questions over long context. "
    "When asked for a secret passphrase, respond with only the passphrase."
)
NEEDLES = [
    "amber raven 417",
    "cobalt cedar 928",
    "solar maple 163",
    "velvet comet 502",
    "silver lantern 741",
    "ember willow 286",
]
FILLER_TEMPLATES = [
    "Paragraph {paragraph}. A field report on coastal weather notes calm winds, cool water, and patient fishing boats anchored at dawn.",
    "Paragraph {paragraph}. An operations memo describes routine maintenance on generators, valves, and backup telemetry links in the control room.",
    "Paragraph {paragraph}. A travel journal mentions narrow streets, warm bread, small museums, and afternoon trains crossing the river valley.",
    "Paragraph {paragraph}. An engineering notebook records test fixtures, revised tolerances, and a checklist for inspecting vibration and heat drift.",
    "Paragraph {paragraph}. A biology summary explains pollination timing, seed dispersal, and the seasonal behavior of insects near flowering shrubs.",
    "Paragraph {paragraph}. A logistics brief lists container arrivals, warehouse bays, forklift routes, and delayed pallets scheduled for evening unload.",
    "Paragraph {paragraph}. A kitchen diary reflects on simmering broth, roasted vegetables, herb storage, and careful knife work before service.",
    "Paragraph {paragraph}. A city planning note reviews bike lanes, storm drains, school crossings, and a survey of traffic near the station.",
    "Paragraph {paragraph}. A geology primer discusses sandstone layers, river erosion, mineral stains, and the shape of canyons after spring rain.",
    "Paragraph {paragraph}. A software handoff document covers deployment windows, monitoring dashboards, error budgets, and rollback practice for weekends.",
]


def make_generation_stats() -> Dict[str, float]:
    return {
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
    }


def build_context(tokenizer, target_tokens: int, depth: float, needle: str) -> str:
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be in [0, 1].")

    paragraphs: List[str] = []
    paragraph_index = 0
    while True:
        candidate_context = " ".join(paragraphs)
        if len(tokenizer.encode(candidate_context)) >= target_tokens:
            break
        template = FILLER_TEMPLATES[paragraph_index % len(FILLER_TEMPLATES)]
        paragraphs.append(template.format(paragraph=paragraph_index + 1))
        paragraph_index += 1

    needle_sentence = (
        f"Paragraph {paragraph_index + 1}. This is the only special record in the document. "
        f"The secret passphrase is {needle}."
    )
    insert_at = min(max(int(round(depth * len(paragraphs))), 0), len(paragraphs))
    paragraphs.insert(insert_at, needle_sentence)
    return "\n".join(paragraphs)


def run_generation(model, tokenizer, prompt_tokens, cache, max_tokens: int, seed: int):
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(seed)
    sampler = make_sampler(temp=0.0)
    started = time.time()
    text = generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        prompt_cache=cache,
        sampler=sampler,
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed = time.time() - started
    stats = make_generation_stats()
    stats["seconds"] = elapsed
    stats["cache_storage_gb"] = bytes_to_gb(cache_nbytes(cache))
    return text, stats


def make_turbo_cache(model, bits: int, seed: int, implementation: str, block_size: int):
    layers = list(get_transformer_layers(model))
    if implementation == "direct":
        return [
            TurboQuantDirectKVCache(bits=bits, seed=seed + i, block_size=block_size)
            for i, _ in enumerate(layers)
        ]
    return [
        TurboQuantKVCache(bits=bits, seed=seed + i, use_dense_shadow=True)
        for i, _ in enumerate(layers)
    ]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description="Run a Needle-In-A-Haystack-style benchmark on a local MLX model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bits", type=int, default=3, help="TurboQuant KV-cache bit-width.")
    parser.add_argument(
        "--implementation",
        choices=["direct", "shadow"],
        default="direct",
        help="TurboQuant cache implementation to benchmark.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--target-tokens",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
        help="Approximate context token targets before the final question prompt.",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Needle insertion depths in the haystack.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(resolved_model)
    enable_turboquant_direct_attention(model)

    total_baseline_contains = 0
    total_turbo_contains = 0
    total_baseline_exact = 0
    total_turbo_exact = 0
    total_cases = 0

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"bits={args.bits}")
    print(f"implementation={args.implementation}")
    print(f"block_size={args.block_size}")

    for case_index, target_tokens in enumerate(args.target_tokens):
        for depth_index, depth in enumerate(args.depths):
            needle = NEEDLES[(case_index * len(args.depths) + depth_index) % len(NEEDLES)]
            context = build_context(tokenizer, target_tokens=target_tokens, depth=depth, needle=needle)
            user_prompt = (
                "Read the long context below and answer the final question.\n\n"
                f"Context:\n{context}\n\n"
                "Question: What is the secret passphrase?\n"
                "Answer with only the passphrase."
            )
            prompt_tokens = encode_chat_prompt(
                tokenizer,
                user_prompt=user_prompt,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )

            baseline_cache = model.make_cache()
            baseline_text, baseline_stats = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                cache=baseline_cache,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            baseline_exact = exact_match(baseline_text, [needle])
            baseline_contains = contains_answer(baseline_text, [needle])

            del baseline_cache
            gc.collect()
            mx.clear_cache()

            turbo_cache = make_turbo_cache(
                model,
                bits=args.bits,
                seed=args.seed,
                implementation=args.implementation,
                block_size=args.block_size,
            )
            turbo_text, turbo_stats = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                cache=turbo_cache,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            turbo_exact = exact_match(turbo_text, [needle])
            turbo_contains = contains_answer(turbo_text, [needle])

            total_baseline_contains += int(baseline_contains)
            total_turbo_contains += int(turbo_contains)
            total_baseline_exact += int(baseline_exact)
            total_turbo_exact += int(turbo_exact)
            total_cases += 1

            print(
                "case "
                f"target_tokens={target_tokens} "
                f"depth={depth:.2f} "
                f"prompt_tokens={len(prompt_tokens)} "
                f"needle={needle!r} "
                f"baseline_contains={int(baseline_contains)} "
                f"turbo_contains={int(turbo_contains)} "
                f"baseline_exact={int(baseline_exact)} "
                f"turbo_exact={int(turbo_exact)} "
                f"baseline_cache_storage_gb={baseline_stats['cache_storage_gb']:.6f} "
                f"turbo_cache_storage_gb={turbo_stats['cache_storage_gb']:.6f} "
                f"baseline_peak_memory_gb={baseline_stats['peak_memory_gb']:.6f} "
                f"turbo_peak_memory_gb={turbo_stats['peak_memory_gb']:.6f} "
                f"baseline_seconds={baseline_stats['seconds']:.2f} "
                f"turbo_seconds={turbo_stats['seconds']:.2f}"
            )
            print(f"baseline_answer={baseline_text.strip()!r}")
            print(f"turbo_answer={turbo_text.strip()!r}")

            del turbo_cache
            gc.collect()
            mx.clear_cache()

    print(f"cases={total_cases}")
    print(f"baseline_exact_accuracy={total_baseline_exact / max(total_cases, 1):.4f}")
    print(f"turbo_exact_accuracy={total_turbo_exact / max(total_cases, 1):.4f}")
    print(f"baseline_contains_accuracy={total_baseline_contains / max(total_cases, 1):.4f}")
    print(f"turbo_contains_accuracy={total_turbo_contains / max(total_cases, 1):.4f}")


if __name__ == "__main__":
    main()
