from __future__ import annotations

import argparse
import gc

import mlx.core as mx
import numpy as np
from mlx_lm import load

from turboquant.hf_cache import resolve_cached_model_path
from turboquant.kv_cache import TurboQuantDirectKVCache, TurboQuantKVCache
from turboquant.mlx_attention import enable_turboquant_direct_attention, get_transformer_layers


DEFAULT_PROMPT = "Vector quantization can compress representations while preserving useful geometry."


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


def run_prefill(model, input_ids: mx.array, cache):
    mx.clear_cache()
    mx.reset_peak_memory()
    logits = np.asarray(model(input_ids, cache=cache), dtype=np.float64)
    stats = {
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
        "cache_storage_gb": bytes_to_gb(sum(c.nbytes for c in cache)),
    }
    return logits, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure real MLX KV-cache storage for baseline vs TurboQuant."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="HF/MLX model repo or local path.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--repeat",
        type=int,
        default=64,
        help="Repeat the prompt text to build a longer context.",
    )
    parser.add_argument("--bits", type=int, default=3, help="TurboQuant bit-width.")
    parser.add_argument(
        "--implementation",
        choices=["direct", "shadow"],
        default="direct",
        help="TurboQuant cache implementation to test.",
    )
    parser.add_argument(
        "--dense-shadow",
        action="store_true",
        help="Keep a decoded MLX shadow cache for speed at the cost of higher runtime memory. Only used for shadow mode.",
    )
    parser.add_argument("--block-size", type=int, default=256, help="Direct-cache block size.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(resolved_model)
    enable_turboquant_direct_attention(model)

    prompt_text = build_prompt_text(args.prompt, args.repeat)
    prompt_tokens = tokenizer.encode(prompt_text)
    input_ids = mx.array([prompt_tokens])

    baseline_cache = model.make_cache()
    baseline_logits, baseline_stats = run_prefill(model, input_ids, baseline_cache)
    baseline_last = baseline_logits[0, -1, :]

    del baseline_cache
    gc.collect()
    mx.clear_cache()

    layers = list(get_transformer_layers(model))
    if args.implementation == "direct":
        turbo_cache = [
            TurboQuantDirectKVCache(
                bits=args.bits,
                seed=args.seed + i,
                compute_stats=True,
                block_size=args.block_size,
            )
            for i, _ in enumerate(layers)
        ]
    else:
        turbo_cache = [
            TurboQuantKVCache(
                bits=args.bits,
                seed=args.seed + i,
                compute_stats=True,
                use_dense_shadow=args.dense_shadow,
            )
            for i, _ in enumerate(layers)
        ]
    turbo_logits, turbo_stats = run_prefill(model, input_ids, turbo_cache)
    turbo_last = turbo_logits[0, -1, :]

    base_probs = stable_softmax(baseline_last)
    turbo_probs = stable_softmax(turbo_last)

    layer_key_cosine = np.mean([c.stats["key_mean_cosine"] for c in turbo_cache])
    layer_value_cosine = np.mean([c.stats["value_mean_cosine"] for c in turbo_cache])
    layer_key_mse = np.mean([c.stats["key_mse"] for c in turbo_cache])
    layer_value_mse = np.mean([c.stats["value_mse"] for c in turbo_cache])

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"bits={args.bits}")
    print(f"implementation={args.implementation}")
    print(f"dense_shadow={int(args.dense_shadow)}")
    print(f"block_size={args.block_size}")
    print(f"prompt_tokens={len(prompt_tokens)}")
    print(f"baseline_cache_storage_gb={baseline_stats['cache_storage_gb']:.6f}")
    print(f"turboquant_cache_storage_gb={turbo_stats['cache_storage_gb']:.6f}")
    print(f"turboquant_dense_shadow_gb={bytes_to_gb(sum(c.dense_nbytes for c in turbo_cache)):.6f}")
    print(f"turboquant_index_shadow_gb={bytes_to_gb(sum(getattr(c, 'index_shadow_nbytes', 0) for c in turbo_cache)):.6f}")
    print(f"storage_compression_ratio={baseline_stats['cache_storage_gb'] / max(turbo_stats['cache_storage_gb'], 1e-12):.4f}")
    print(f"baseline_peak_memory_gb={baseline_stats['peak_memory_gb']:.6f}")
    print(f"turboquant_peak_memory_gb={turbo_stats['peak_memory_gb']:.6f}")
    print(f"baseline_active_memory_gb={baseline_stats['active_memory_gb']:.6f}")
    print(f"turboquant_active_memory_gb={turbo_stats['active_memory_gb']:.6f}")
    print(f"logit_top1_match={1.0 if int(np.argmax(baseline_last)) == int(np.argmax(turbo_last)) else 0.0:.4f}")
    print(f"logit_top5_overlap={topk_overlap(baseline_last, turbo_last, 5):.4f}")
    print(f"logit_top10_overlap={topk_overlap(baseline_last, turbo_last, 10):.4f}")
    print(f"logit_l2={np.linalg.norm(baseline_last - turbo_last):.4f}")
    print(f"prob_l1={np.sum(np.abs(base_probs - turbo_probs)):.4f}")
    print(f"mean_key_cosine={layer_key_cosine:.4f}")
    print(f"mean_value_cosine={layer_value_cosine:.4f}")
    print(f"mean_key_mse={layer_key_mse:.6f}")
    print(f"mean_value_mse={layer_value_mse:.6f}")


if __name__ == "__main__":
    main()
