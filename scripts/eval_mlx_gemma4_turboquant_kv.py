from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from turboquant.hf_cache import resolve_cached_model_path
from turboquant.kv_cache import (
    TurboQuantDirectKVCache,
    TurboQuantKVCache,
    TurboQuantQuantizerPool,
    cache_packed_nbytes,
)
from turboquant.mlx_attention import enable_turboquant_gemma4_attention


DEFAULT_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
DEFAULT_PROMPT = (
    "Explain why KV-cache compression matters for long-context language model serving "
    "in one short paragraph."
)


def gemma4_args(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "args"):
        return model.language_model.args
    return model.args


def bytes_to_gb(value: int) -> float:
    return float(value) / 1e9


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    topk_a = set(np.argsort(-a)[:k].tolist())
    topk_b = set(np.argsort(-b)[:k].tolist())
    return len(topk_a & topk_b) / float(k)


def build_prompt_text(prompt: str, repeat: int) -> str:
    return " ".join([prompt] * repeat)


def encode_chat_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(rendered)
    return tokenizer.encode(prompt)


def run_prefill(model, input_ids: mx.array, cache):
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.time()
    logits = model(input_ids, cache=cache)
    last_logits = logits[0, -1, :].astype(mx.float32)
    mx.eval(last_logits)
    elapsed = time.time() - started
    stats = {
        "seconds": elapsed,
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
    }
    return np.array(last_logits.tolist(), dtype=np.float64), stats


def run_generation(
    model,
    tokenizer,
    prompt_tokens: list[int],
    cache,
    *,
    max_tokens: int,
    temp: float,
    prefill_step_size: int,
):
    mx.clear_cache()
    mx.reset_peak_memory()
    sampler = make_sampler(temp=temp)
    started = time.time()
    text = generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        prompt_cache=cache,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
        prefill_step_size=prefill_step_size,
    )
    elapsed = time.time() - started
    stats = {
        "seconds": elapsed,
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
    }
    return text, stats


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
                        compute_stats=True,
                        block_size=block_size,
                        recent_window_tokens=recent_window,
                        recent_slack_tokens=recent_slack,
                        quantizer_pool=quantizer_pool,
                    )
                )
            else:
                caches.append(
                    TurboQuantKVCache(
                        bits=bits,
                        seed=seed + i,
                        compute_stats=True,
                        use_dense_shadow=dense_shadow,
                        recent_window_tokens=recent_window,
                        recent_slack_tokens=recent_slack,
                        quantizer_pool=quantizer_pool,
                    )
                )
        else:
            caches.append(
                RotatingKVCache(
                    max_size=args.sliding_window,
                    keep=0,
                )
            )
    return caches


def cache_breakdown(model, cache) -> dict[str, float]:
    args = gemma4_args(model)
    first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers
    layer_types = args.layer_types
    total = sum(c.nbytes for c in cache)
    full_attention = 0
    sliding = 0
    packed_only = 0
    quantizer_metadata = 0
    dense_shadow = 0
    index_shadow = 0
    for i in range(first_kv_shared):
        entry = cache[i]
        if layer_types[i] == "full_attention":
            full_attention += entry.nbytes
            packed_only += getattr(entry, "nbytes", 0) - getattr(entry, "dense_nbytes", 0)
            quantizer_metadata += getattr(entry, "_quantizer_pool", None) is None and 0 or 0
            dense_shadow += getattr(entry, "dense_nbytes", 0)
            index_shadow += getattr(entry, "index_shadow_nbytes", 0)
        else:
            sliding += entry.nbytes

    metadata_seen = set()
    metadata_total = 0
    for entry in cache:
        pool = getattr(entry, "_quantizer_pool", None)
        if pool is None:
            quantizer_metadata = getattr(entry, "_quantizers", None)
            if quantizer_metadata:
                for q in quantizer_metadata.values():
                    metadata_total += int(getattr(q, "metadata_nbytes", 0))
            continue
        pool_id = id(pool)
        if pool_id in metadata_seen:
            continue
        metadata_seen.add(pool_id)
        for q in pool._quantizers.values():
            metadata_total += int(getattr(q, "metadata_nbytes", 0))

    return {
        "total_gb": bytes_to_gb(total),
        "full_attention_gb": bytes_to_gb(full_attention),
        "sliding_gb": bytes_to_gb(sliding),
        "packed_storage_gb": bytes_to_gb(
            sum(
                cache_packed_nbytes(c)
                for c in cache
                if isinstance(c, (TurboQuantKVCache, TurboQuantDirectKVCache))
            )
        ),
        "quantizer_metadata_gb": bytes_to_gb(metadata_total),
        "dense_shadow_gb": bytes_to_gb(dense_shadow),
        "index_shadow_gb": bytes_to_gb(index_shadow),
    }


def mean_cache_stat(cache, key: str) -> float:
    stats = [c.stats[key] for c in cache if isinstance(c, (TurboQuantKVCache, TurboQuantDirectKVCache))]
    if not stats:
        return 0.0
    return float(np.mean(stats))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Gemma 4 baseline vs TurboQuant KV cache behavior on MLX."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--bits", type=float, default=3.5)
    parser.add_argument(
        "--implementation",
        choices=["shadow", "direct"],
        default="shadow",
    )
    parser.add_argument("--dense-shadow", action="store_true")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--recent-slack", type=int, default=8)
    parser.add_argument(
        "--share-quantizers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(
        resolved_model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
    )

    prompt_text = build_prompt_text(args.prompt, args.repeat)
    prompt_tokens = encode_chat_prompt(tokenizer, prompt_text)
    input_ids = mx.array([prompt_tokens])

    baseline_cache = model.make_cache()
    baseline_logits, baseline_prefill = run_prefill(model, input_ids, baseline_cache)
    baseline_breakdown = cache_breakdown(model, baseline_cache)
    baseline_last = baseline_logits

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
    )
    turbo_logits, turbo_prefill = run_prefill(model, input_ids, turbo_cache)
    turbo_breakdown = cache_breakdown(model, turbo_cache)
    turbo_last = turbo_logits

    baseline_probs = stable_softmax(baseline_last)
    turbo_probs = stable_softmax(turbo_last)

    baseline_text, baseline_generation = run_generation(
        model,
        tokenizer,
        prompt_tokens,
        model.make_cache(),
        max_tokens=args.max_tokens,
        temp=args.temp,
        prefill_step_size=args.prefill_step_size,
    )
    turbo_text, turbo_generation = run_generation(
        model,
        tokenizer,
        prompt_tokens,
        build_gemma4_cache(
            model,
            implementation=args.implementation,
            bits=args.bits,
            seed=args.seed,
            block_size=args.block_size,
            recent_window=args.recent_window,
            recent_slack=args.recent_slack,
            dense_shadow=args.dense_shadow,
            share_quantizers=args.share_quantizers,
        ),
        max_tokens=args.max_tokens,
        temp=args.temp,
        prefill_step_size=args.prefill_step_size,
    )

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"bits={args.bits}")
    print(f"implementation={args.implementation}")
    print(f"dense_shadow={int(args.dense_shadow)}")
    print(f"block_size={args.block_size}")
    print(f"recent_window={args.recent_window}")
    print(f"recent_slack={args.recent_slack}")
    print(f"prompt_tokens={len(prompt_tokens)}")
    print(f"baseline_prefill_seconds={baseline_prefill['seconds']:.2f}")
    print(f"turboquant_prefill_seconds={turbo_prefill['seconds']:.2f}")
    print(f"baseline_cache_storage_gb={baseline_breakdown['total_gb']:.6f}")
    print(f"turboquant_cache_storage_gb={turbo_breakdown['total_gb']:.6f}")
    print(f"baseline_full_attention_gb={baseline_breakdown['full_attention_gb']:.6f}")
    print(f"baseline_sliding_gb={baseline_breakdown['sliding_gb']:.6f}")
    print(f"turboquant_full_attention_gb={turbo_breakdown['full_attention_gb']:.6f}")
    print(f"turboquant_sliding_gb={turbo_breakdown['sliding_gb']:.6f}")
    print(f"turboquant_packed_storage_gb={turbo_breakdown['packed_storage_gb']:.6f}")
    print(f"turboquant_quantizer_metadata_gb={turbo_breakdown['quantizer_metadata_gb']:.6f}")
    print(f"turboquant_dense_shadow_gb={turbo_breakdown['dense_shadow_gb']:.6f}")
    print(f"turboquant_index_shadow_gb={turbo_breakdown['index_shadow_gb']:.6f}")
    print(f"storage_compression_ratio={baseline_breakdown['total_gb'] / max(turbo_breakdown['total_gb'], 1e-12):.4f}")
    print(f"full_attention_compression_ratio={baseline_breakdown['full_attention_gb'] / max(turbo_breakdown['full_attention_gb'], 1e-12):.4f}")
    print(f"baseline_peak_memory_gb={baseline_prefill['peak_memory_gb']:.6f}")
    print(f"turboquant_peak_memory_gb={turbo_prefill['peak_memory_gb']:.6f}")
    print(f"baseline_active_memory_gb={baseline_prefill['active_memory_gb']:.6f}")
    print(f"turboquant_active_memory_gb={turbo_prefill['active_memory_gb']:.6f}")
    print(f"logit_top1_match={1.0 if int(np.argmax(baseline_last)) == int(np.argmax(turbo_last)) else 0.0:.4f}")
    print(f"logit_top5_overlap={topk_overlap(baseline_last, turbo_last, 5):.4f}")
    print(f"logit_top10_overlap={topk_overlap(baseline_last, turbo_last, 10):.4f}")
    print(f"logit_l2={np.linalg.norm(baseline_last - turbo_last):.4f}")
    print(f"prob_l1={np.sum(np.abs(baseline_probs - turbo_probs)):.4f}")
    print(f"mean_key_cosine={mean_cache_stat(turbo_cache, 'key_mean_cosine'):.4f}")
    print(f"mean_value_cosine={mean_cache_stat(turbo_cache, 'value_mean_cosine'):.4f}")
    print(f"mean_key_mse={mean_cache_stat(turbo_cache, 'key_mse'):.6f}")
    print(f"mean_value_mse={mean_cache_stat(turbo_cache, 'value_mse'):.6f}")
    print(f"baseline_generation_seconds={baseline_generation['seconds']:.2f}")
    print(f"turboquant_generation_seconds={turbo_generation['seconds']:.2f}")
    print(f"generation_exact_match={int(baseline_text.strip() == turbo_text.strip())}")
    print("--- baseline_output ---")
    print(baseline_text)
    print("--- turboquant_output ---")
    print(turbo_text)


if __name__ == "__main__":
    main()
