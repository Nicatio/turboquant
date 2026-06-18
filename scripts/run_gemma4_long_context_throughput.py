from __future__ import annotations

import argparse
import gc
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from turboquant.hf_cache import resolve_cached_model_path
from turboquant.kv_cache import (
    TurboQuantDirectKVCache,
    TurboQuantKVCache,
    TurboQuantQuantizerPool,
    cache_list_nbytes,
)
from turboquant.mlx_attention import enable_turboquant_gemma4_attention


DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
DEFAULT_PROMPT = (
    "Explain why KV-cache compression matters for long-context language model serving "
    "in one short paragraph."
)


def gemma4_args(model: Any):
    if hasattr(model, "language_model") and hasattr(model.language_model, "args"):
        return model.language_model.args
    return model.args


def bytes_to_gb(value: int | float) -> float:
    return float(value) / 1e9


def cache_metric_gb(cache_entries, attr: str) -> float:
    return bytes_to_gb(sum(getattr(entry, attr, 0) for entry in cache_entries))


def build_prompt_text(prompt: str, repeat: int) -> str:
    return " ".join([prompt] * repeat)


def encode_chat_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer.encode(rendered)
    return tokenizer.encode(prompt)


def find_repeat_for_target(tokenizer, prompt: str, target_prompt_tokens: int) -> tuple[int, int]:
    low = 1
    high = 1
    while True:
        tokens = len(encode_chat_prompt(tokenizer, build_prompt_text(prompt, high)))
        if tokens >= target_prompt_tokens:
            break
        low = high
        high *= 2

    best_repeat = low
    best_tokens = len(encode_chat_prompt(tokenizer, build_prompt_text(prompt, low)))
    left = low
    right = high
    while left <= right:
        mid = (left + right) // 2
        tokens = len(encode_chat_prompt(tokenizer, build_prompt_text(prompt, mid)))
        if tokens <= target_prompt_tokens:
            best_repeat = mid
            best_tokens = tokens
            left = mid + 1
        else:
            right = mid - 1
    return best_repeat, best_tokens


def build_gemma4_cache(
    model,
    *,
    implementation: str,
    bits: float,
    seed: int,
    block_size: int,
    recent_window: int,
    recent_slack: int,
    dense_shadow: bool,
    share_quantizers: bool,
    lean_direct: bool,
    metal_max_query_length: int,
):
    args = gemma4_args(model)
    first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers
    layer_types = args.layer_types
    quantizer_pool = TurboQuantQuantizerPool() if share_quantizers else None
    caches = []
    for i in range(first_kv_shared):
        if layer_types[i] == "full_attention":
            if implementation == "direct":
                caches.append(
                    TurboQuantDirectKVCache(
                        bits=bits,
                        seed=seed + i,
                        compute_stats=False,
                        block_size=block_size,
                        recent_window_tokens=recent_window,
                        recent_slack_tokens=recent_slack,
                        quantizer_pool=quantizer_pool,
                        lean_mode=lean_direct,
                        metal_max_query_length=metal_max_query_length,
                    )
                )
            else:
                caches.append(
                    TurboQuantKVCache(
                        bits=bits,
                        seed=seed + i,
                        compute_stats=False,
                        use_dense_shadow=dense_shadow,
                        recent_window_tokens=recent_window,
                        recent_slack_tokens=recent_slack,
                        quantizer_pool=quantizer_pool,
                    )
                )
        else:
            caches.append(RotatingKVCache(max_size=args.sliding_window, keep=0))
    return caches


def build_gemma4_baseline_cache(
    model,
    *,
    cache_mode: str,
    quant_bits: int,
    quant_group_size: int,
):
    if cache_mode == "default":
        return model.make_cache()

    if cache_mode != "q8":
        raise ValueError(f"Unsupported baseline cache mode: {cache_mode}")

    args = gemma4_args(model)
    first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers
    layer_types = args.layer_types
    caches = []
    for i in range(first_kv_shared):
        if layer_types[i] == "full_attention":
            caches.append(
                QuantizedKVCache(
                    group_size=quant_group_size,
                    bits=quant_bits,
                )
            )
        else:
            caches.append(RotatingKVCache(max_size=args.sliding_window, keep=0))
    return caches


def run_stream_benchmark(
    model,
    tokenizer,
    prompt_tokens: list[int],
    *,
    cache,
    max_tokens: int,
    temp: float,
    seed: int,
    prefill_step_size: int,
):
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(seed)
    sampler = make_sampler(temp=temp)

    text_parts: list[str] = []
    last_response = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=cache,
        prefill_step_size=prefill_step_size,
    ):
        if response.text:
            text_parts.append(response.text)
        last_response = response

    if last_response is None:
        raise RuntimeError("Generation did not produce a response.")

    return {
        "text": "".join(text_parts),
        "prompt_tokens": int(last_response.prompt_tokens),
        "prompt_tps": float(last_response.prompt_tps),
        "generation_tokens": int(last_response.generation_tokens),
        "generation_tps": float(last_response.generation_tps),
        "peak_memory_gb": float(last_response.peak_memory),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
        "cache_storage_gb": bytes_to_gb(cache_list_nbytes(cache)),
        "finish_reason": last_response.finish_reason or "unknown",
    }


def print_stats(prefix: str, stats: dict[str, Any]) -> None:
    print(f"{prefix}_prompt_tokens={stats['prompt_tokens']}")
    print(f"{prefix}_prompt_tps={stats['prompt_tps']:.3f}")
    print(f"{prefix}_generation_tokens={stats['generation_tokens']}")
    print(f"{prefix}_generation_tps={stats['generation_tps']:.3f}")
    print(f"{prefix}_peak_memory_gb={stats['peak_memory_gb']:.3f}")
    print(f"{prefix}_active_memory_gb={stats['active_memory_gb']:.3f}")
    print(f"{prefix}_cache_memory_gb={stats['cache_memory_gb']:.3f}")
    print(f"{prefix}_cache_storage_gb={stats['cache_storage_gb']:.6f}")
    print(f"{prefix}_finish_reason={stats['finish_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Gemma 4 long-context prompt/generation throughput on MLX."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--target-prompt-tokens", type=int, default=128000)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument(
        "--implementation",
        choices=["baseline", "shadow", "direct"],
        default="shadow",
    )
    parser.add_argument(
        "--run-mode",
        choices=["both", "baseline-only", "turbo-only"],
        default="both",
        help="Run baseline, TurboQuant, or both sequentially in one process.",
    )
    parser.add_argument(
        "--baseline-cache-mode",
        choices=["default", "q8"],
        default="default",
        help="Baseline cache implementation. 'q8' uses MLX QuantizedKVCache(bits=8) on full-attention layers.",
    )
    parser.add_argument(
        "--baseline-quant-bits",
        type=int,
        default=8,
        help="Bit-width for the q8 baseline QuantizedKVCache.",
    )
    parser.add_argument(
        "--baseline-quant-group-size",
        type=int,
        default=64,
        help="Group size for the q8 baseline QuantizedKVCache.",
    )
    parser.add_argument("--bits", type=float, default=3.5)
    parser.add_argument("--dense-shadow", action="store_true")
    parser.add_argument(
        "--lean-direct",
        action="store_true",
        help="Disable direct-path index shadow, compressed prefix shadow, and rotated block caches.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--recent-slack", type=int, default=8)
    parser.add_argument(
        "--share-quantizers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument(
        "--metal-max-query-length",
        type=int,
        default=1,
        help="Maximum query length that the direct path will route through Metal fused score/value kernels.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(
        resolved_model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    if args.repeat is None:
        repeat, calibrated_tokens = find_repeat_for_target(
            tokenizer,
            args.prompt,
            args.target_prompt_tokens,
        )
    else:
        repeat = args.repeat
        calibrated_tokens = len(encode_chat_prompt(tokenizer, build_prompt_text(args.prompt, repeat)))

    prompt_text = build_prompt_text(args.prompt, repeat)
    prompt_tokens = encode_chat_prompt(tokenizer, prompt_text)

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"implementation={args.implementation}")
    print(f"lean_direct={int(args.lean_direct)}")
    print(f"run_mode={args.run_mode}")
    print(f"baseline_cache_mode={args.baseline_cache_mode}")
    print(f"baseline_quant_bits={args.baseline_quant_bits}")
    print(f"baseline_quant_group_size={args.baseline_quant_group_size}")
    print(f"bits={args.bits}")
    print(f"repeat={repeat}")
    print(f"target_prompt_tokens={args.target_prompt_tokens}")
    print(f"calibrated_prompt_tokens={calibrated_tokens}")
    print(f"actual_prompt_tokens={len(prompt_tokens)}")
    print(f"max_tokens={args.max_tokens}")
    print(f"prefill_step_size={args.prefill_step_size}")
    print(f"metal_max_query_length={args.metal_max_query_length}")

    if args.run_mode in ("both", "baseline-only") or args.implementation == "baseline":
        baseline_cache = build_gemma4_baseline_cache(
            model,
            cache_mode=args.baseline_cache_mode,
            quant_bits=args.baseline_quant_bits,
            quant_group_size=args.baseline_quant_group_size,
        )
        baseline_stats = run_stream_benchmark(
            model,
            tokenizer,
            prompt_tokens,
            cache=baseline_cache,
            max_tokens=args.max_tokens,
            temp=args.temp,
            seed=args.seed,
            prefill_step_size=args.prefill_step_size,
        )
        print_stats("baseline", baseline_stats)
    else:
        baseline_stats = None

    if args.implementation == "baseline" or args.run_mode == "baseline-only":
        if baseline_stats is not None:
            print("--- baseline_output ---")
            print(baseline_stats["text"])
        return

    if baseline_stats is not None:
        del baseline_cache
        gc.collect()
        mx.clear_cache()

    if args.implementation == "direct":
        enable_turboquant_gemma4_attention(model)

    turbo_cache = build_gemma4_cache(
        model,
        implementation=args.implementation,
        bits=args.bits,
        seed=args.seed,
        block_size=args.block_size,
        recent_window=args.recent_window,
        recent_slack=args.recent_slack,
        dense_shadow=args.dense_shadow,
        share_quantizers=args.share_quantizers,
        lean_direct=args.lean_direct,
        metal_max_query_length=args.metal_max_query_length,
    )
    turbo_stats = run_stream_benchmark(
        model,
        tokenizer,
        prompt_tokens,
        cache=turbo_cache,
        max_tokens=args.max_tokens,
        temp=args.temp,
        seed=args.seed,
        prefill_step_size=args.prefill_step_size,
    )
    print_stats("turboquant", turbo_stats)
    print(
        f"turboquant_dense_shadow_gb={cache_metric_gb(turbo_cache, 'dense_nbytes'):.6f}"
    )
    print(
        f"turboquant_index_shadow_gb={cache_metric_gb(turbo_cache, 'index_shadow_nbytes'):.6f}"
    )
    print(
        f"turboquant_prefix_shadow_gb={cache_metric_gb(turbo_cache, 'prefix_shadow_nbytes'):.6f}"
    )
    print(
        "turboquant_rotated_block_cache_gb="
        f"{cache_metric_gb(turbo_cache, 'rotated_block_cache_nbytes'):.6f}"
    )
    if baseline_stats is not None:
        print(
            "prefill_ratio="
            f"{turbo_stats['prompt_tps'] / max(baseline_stats['prompt_tps'], 1e-12):.4f}"
        )
        print(
            "generation_ratio="
            f"{turbo_stats['generation_tps'] / max(baseline_stats['generation_tps'], 1e-12):.4f}"
        )
        print(
            "storage_compression_ratio="
            f"{baseline_stats['cache_storage_gb'] / max(turbo_stats['cache_storage_gb'], 1e-12):.4f}"
        )
        print(
            f"generation_exact_match={int(baseline_stats['text'].strip() == turbo_stats['text'].strip())}"
        )
        print("--- baseline_output ---")
        print(baseline_stats["text"])
    print("--- turboquant_output ---")
    print(turbo_stats["text"])


if __name__ == "__main__":
    main()
