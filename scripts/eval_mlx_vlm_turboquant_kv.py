from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vlm import apply_chat_template, generate, prepare_inputs
from mlx_vlm.models.cache import ArraysCache
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from turboquant.kv_cache import (
    TurboQuantDirectKVCache,
    TurboQuantKVCache,
    TurboQuantQuantizerPool,
    cache_list_nbytes,
    cache_packed_nbytes,
    cache_quantizer_metadata_nbytes,
    quantizer_metadata_nbytes,
)
from turboquant.mlx_attention import (
    enable_turboquant_qwen_vlm_attention,
    get_transformer_layers,
)
from turboquant.mlx_vlm_utils import load_with_slow_processor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "assets" / "sample_grid.ppm"
DEFAULT_PROMPT = (
    "What colors appear in the four quadrants of this image? "
    "Answer with only the colors from top-left to bottom-right."
)


def bytes_to_gb(value: int) -> float:
    return float(value) / 1e9


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def build_prompt_text(prompt: str, repeat: int) -> str:
    return " ".join([prompt] * repeat)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    topk_a = set(np.argsort(-a)[:k].tolist())
    topk_b = set(np.argsort(-b)[:k].tolist())
    return len(topk_a & topk_b) / float(k)


@dataclass
class CacheBreakdown:
    total_gb: float
    linear_state_gb: float
    full_attention_gb: float
    packed_storage_gb: float
    quantizer_metadata_gb: float


def cache_storage_gb(cache) -> float:
    return bytes_to_gb(cache_list_nbytes(cache))


def dense_shadow_gb(cache) -> float:
    return bytes_to_gb(sum(getattr(c, "dense_nbytes", 0) for c in cache))


def index_shadow_gb(cache) -> float:
    return bytes_to_gb(sum(getattr(c, "index_shadow_nbytes", 0) for c in cache))


def prefix_shadow_gb(cache) -> float:
    return bytes_to_gb(sum(getattr(c, "prefix_shadow_nbytes", 0) for c in cache))


def rotated_block_cache_gb(cache) -> float:
    return bytes_to_gb(sum(getattr(c, "rotated_block_cache_nbytes", 0) for c in cache))


def mean_stat(cache, key: str) -> float:
    values = [c.stats[key] for c in cache if hasattr(c, "stats")]
    if not values:
        return 0.0
    return float(np.mean(values))


def get_cache_breakdown(cache) -> CacheBreakdown:
    total_bytes = cache_list_nbytes(cache)
    linear_state_bytes = sum(
        getattr(c, "nbytes", 0) for c in cache if isinstance(c, ArraysCache)
    )
    full_attention_caches = [c for c in cache if not isinstance(c, ArraysCache)]
    packed_storage_bytes = sum(cache_packed_nbytes(entry) for entry in full_attention_caches)
    quantizer_metadata_bytes = sum(
        cache_quantizer_metadata_nbytes(entry)
        for entry in full_attention_caches
        if isinstance(entry, (TurboQuantDirectKVCache, TurboQuantKVCache))
        and entry._quantizer_pool is None
    )
    seen_pools = set()
    for entry in full_attention_caches:
        if not isinstance(entry, (TurboQuantDirectKVCache, TurboQuantKVCache)):
            continue
        pool = entry._quantizer_pool
        if pool is None:
            continue
        pool_id = id(pool)
        if pool_id in seen_pools:
            continue
        seen_pools.add(pool_id)
        quantizer_metadata_bytes += quantizer_metadata_nbytes(pool._quantizers)
    full_attention_bytes = packed_storage_bytes + quantizer_metadata_bytes

    return CacheBreakdown(
        total_gb=bytes_to_gb(total_bytes),
        linear_state_gb=bytes_to_gb(linear_state_bytes),
        full_attention_gb=bytes_to_gb(full_attention_bytes),
        packed_storage_gb=bytes_to_gb(packed_storage_bytes),
        quantizer_metadata_gb=bytes_to_gb(quantizer_metadata_bytes),
    )


def build_qwen_cache(
    model,
    *,
    implementation: str,
    bits: float,
    seed: int,
    block_size: int,
    recent_window: int,
    recent_slack: int,
    dense_shadow: bool,
    compute_stats: bool,
    share_quantizers: bool,
    lean_direct: bool,
    metal_max_query_length: int,
    adaptive_tail_layers: int = 0,
    adaptive_tail_quant_bits: int = 8,
    adaptive_tail_group_size: int = 64,
):
    quantizer_pool = TurboQuantQuantizerPool() if share_quantizers else None
    cache = []
    layers = list(get_transformer_layers(model))
    full_attention_layers = [
        idx for idx, layer in enumerate(layers) if not getattr(layer, "is_linear", False)
    ]
    adaptive_tail_layers = max(0, min(adaptive_tail_layers, len(full_attention_layers)))
    adaptive_cutoff = len(full_attention_layers) - adaptive_tail_layers
    full_attention_index = 0
    for i, layer in enumerate(layers):
        if getattr(layer, "is_linear", False):
            cache.append(ArraysCache(size=2))
            continue

        if full_attention_index >= adaptive_cutoff:
            if adaptive_tail_quant_bits > 0:
                cache.append(
                    QuantizedKVCache(
                        group_size=adaptive_tail_group_size,
                        bits=adaptive_tail_quant_bits,
                    )
                )
            else:
                cache.append(KVCache())
            full_attention_index += 1
            continue

        layer_seed = seed if share_quantizers else seed + i
        if implementation == "direct":
            cache.append(
                TurboQuantDirectKVCache(
                    bits=bits,
                    seed=layer_seed,
                    compute_stats=compute_stats,
                    block_size=block_size,
                    recent_window_tokens=recent_window,
                    recent_slack_tokens=recent_slack,
                    quantizer_pool=quantizer_pool,
                    lean_mode=lean_direct,
                    metal_max_query_length=metal_max_query_length,
                )
            )
        else:
            cache.append(
                TurboQuantKVCache(
                    bits=bits,
                    seed=layer_seed,
                    compute_stats=compute_stats,
                    use_dense_shadow=dense_shadow,
                    recent_window_tokens=recent_window,
                    recent_slack_tokens=recent_slack,
                    quantizer_pool=quantizer_pool,
                )
            )
        full_attention_index += 1
    return cache


def build_qwen_baseline_cache(
    model,
    *,
    cache_mode: str,
    quant_bits: int,
    quant_group_size: int,
):
    if cache_mode == "default":
        if hasattr(model, "make_cache"):
            return model.make_cache()
        if hasattr(model, "language_model") and hasattr(model.language_model, "make_cache"):
            return model.language_model.make_cache()
        cache = []
        for layer in get_transformer_layers(model):
            if getattr(layer, "is_linear", False):
                cache.append(ArraysCache(size=2))
            else:
                cache.append(KVCache())
        return cache

    if cache_mode != "q8":
        raise ValueError(f"Unsupported baseline cache mode: {cache_mode}")

    cache = []
    for layer in get_transformer_layers(model):
        if getattr(layer, "is_linear", False):
            cache.append(ArraysCache(size=2))
        else:
            cache.append(
                QuantizedKVCache(
                    group_size=quant_group_size,
                    bits=quant_bits,
                )
            )
    return cache


def prepare_multimodal_inputs(model, processor, prompt: str, image: str):
    templated_prompt = apply_chat_template(
        processor,
        model.config,
        prompt,
        num_images=1 if image else 0,
    )
    inputs = prepare_inputs(
        processor,
        images=image if image else None,
        prompts=templated_prompt,
        return_tensors="mlx",
    )
    return templated_prompt, inputs


def run_prefill(model, inputs: dict, cache):
    model_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "pixel_values"} and value is not None
    }
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.time()
    outputs = model(
        inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        mask=inputs.get("attention_mask"),
        cache=cache,
        **model_kwargs,
    )
    last_logits = outputs.logits[:, -1, :].astype(mx.float32)
    mx.eval(last_logits)
    finished = time.time()
    stats = {
        "seconds": finished - started,
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
        "cache_storage_gb": cache_storage_gb(cache),
    }
    return np.asarray(last_logits.tolist(), dtype=np.float64), stats


def run_generation(
    model,
    processor,
    *,
    prompt: str,
    image: str,
    cache,
    max_tokens: int,
    temperature: float,
    prefill_step_size: int,
):
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.time()
    result = generate(
        model,
        processor,
        prompt=prompt,
        image=image,
        prompt_cache=cache,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
        prefill_step_size=prefill_step_size,
    )
    finished = time.time()
    stats = {
        "seconds": finished - started,
        "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
        "prompt_tps": float(getattr(result, "prompt_tps", 0.0) or 0.0),
        "generation_tokens": int(getattr(result, "generation_tokens", max_tokens) or 0),
        "generation_tps": float(getattr(result, "generation_tps", 0.0) or 0.0),
        "peak_memory_gb": max(result.peak_memory, bytes_to_gb(mx.get_peak_memory())),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
        "cache_storage_gb": cache_storage_gb(cache),
        "finish_reason": getattr(result, "finish_reason", None) or "unknown",
    }
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs TurboQuant KV-cache behavior on Qwen multimodal MLX models."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-35B-A3B-4bit",
        help="HF/MLX VLM model repo or local path.",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Path or URL to an image.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the prompt text to create a longer multimodal context.",
    )
    parser.add_argument("--bits", type=float, default=3.0)
    parser.add_argument(
        "--implementation",
        choices=["direct", "shadow"],
        default="direct",
        help="TurboQuant cache implementation to test on full-attention layers.",
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
    parser.add_argument(
        "--dense-shadow",
        action="store_true",
        help="Keep decoded dense caches for the shadow implementation.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--recent-window",
        type=int,
        default=0,
        help="Keep this many newest full-attention tokens dense for speed before compressing overflow.",
    )
    parser.add_argument(
        "--recent-slack",
        type=int,
        default=0,
        help="Allow this many extra single-token decode steps to stay dense before flushing recent overflow.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lean-direct",
        action="store_true",
        help="Disable direct-path index shadow, compressed prefix shadow, and rotated block caches.",
    )
    parser.add_argument(
        "--share-quantizers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share one TurboQuant rotation/codebook per compatible full-attention head dimension.",
    )
    parser.add_argument(
        "--adaptive-tail-layers",
        type=int,
        default=0,
        help="Keep the last N full-attention layers on an MLX baseline cache instead of TurboQuant.",
    )
    parser.add_argument(
        "--adaptive-tail-quant-bits",
        type=int,
        default=8,
        help="If adaptive tail layers are enabled, use this MLX QuantizedKVCache bit-width. Use 0 for plain KVCache.",
    )
    parser.add_argument(
        "--adaptive-tail-group-size",
        type=int,
        default=64,
        help="Group size for adaptive-tail QuantizedKVCache layers.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument(
        "--metal-max-query-length",
        type=int,
        default=1,
        help="Maximum query length that the direct path will route through Metal fused score/value kernels.",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Skip direct full-sequence logits prefill and measure long-context throughput with chunked generate() only.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    image_arg = str(Path(args.image).expanduser()) if "://" not in args.image else args.image

    resolved_model, model, processor = load_with_slow_processor(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    enable_turboquant_qwen_vlm_attention(model)

    prompt_text = build_prompt_text(args.prompt, args.repeat)
    templated_prompt, inputs = prepare_multimodal_inputs(
        model,
        processor,
        prompt_text,
        image_arg,
    )

    baseline_prefill = None
    baseline_last = None
    baseline_breakdown = None
    baseline_generation = None
    baseline_generation_stats = None
    if args.run_mode in ("both", "baseline-only"):
        baseline_cache = build_qwen_baseline_cache(
            model,
            cache_mode=args.baseline_cache_mode,
            quant_bits=args.baseline_quant_bits,
            quant_group_size=args.baseline_quant_group_size,
        )
        if args.throughput_only:
            baseline_generation, baseline_generation_stats = run_generation(
                model,
                processor,
                prompt=templated_prompt,
                image=image_arg,
                cache=baseline_cache,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prefill_step_size=args.prefill_step_size,
            )
            baseline_prefill = baseline_generation_stats
            baseline_breakdown = get_cache_breakdown(baseline_cache)
        else:
            baseline_logits, baseline_prefill = run_prefill(model, inputs, baseline_cache)
            baseline_last = baseline_logits[0]
            baseline_breakdown = get_cache_breakdown(baseline_cache)

        if args.max_tokens > 0 and not args.throughput_only:
            del baseline_cache
            gc.collect()
            mx.clear_cache()
            baseline_cache = build_qwen_baseline_cache(
                model,
                cache_mode=args.baseline_cache_mode,
                quant_bits=args.baseline_quant_bits,
                quant_group_size=args.baseline_quant_group_size,
            )
            baseline_generation, baseline_generation_stats = run_generation(
                model,
                processor,
                prompt=templated_prompt,
                image=image_arg,
                cache=baseline_cache,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prefill_step_size=args.prefill_step_size,
            )

        del baseline_cache
        gc.collect()
        mx.clear_cache()

    turbo_prefill = None
    turbo_last = None
    turbo_breakdown = None
    turbo_dense_shadow_prefill = 0.0
    turbo_index_shadow_prefill = 0.0
    turbo_prefix_shadow_prefill = 0.0
    turbo_rotated_block_prefill = 0.0
    turbo_key_cosine = 0.0
    turbo_value_cosine = 0.0
    turbo_key_mse = 0.0
    turbo_value_mse = 0.0
    turbo_generation = None
    turbo_generation_stats = None

    if args.run_mode in ("both", "turbo-only"):
        turbo_cache = build_qwen_cache(
            model,
            implementation=args.implementation,
            bits=args.bits,
            seed=args.seed,
            block_size=args.block_size,
            recent_window=args.recent_window,
            recent_slack=args.recent_slack,
            dense_shadow=args.dense_shadow,
            compute_stats=True,
        share_quantizers=args.share_quantizers,
        lean_direct=args.lean_direct,
        metal_max_query_length=args.metal_max_query_length,
        adaptive_tail_layers=args.adaptive_tail_layers,
            adaptive_tail_quant_bits=args.adaptive_tail_quant_bits,
            adaptive_tail_group_size=args.adaptive_tail_group_size,
        )
        if args.throughput_only:
            turbo_generation, turbo_generation_stats = run_generation(
                model,
                processor,
                prompt=templated_prompt,
                image=image_arg,
                cache=turbo_cache,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prefill_step_size=args.prefill_step_size,
            )
            turbo_prefill = turbo_generation_stats
            turbo_breakdown = get_cache_breakdown(turbo_cache)
            turbo_dense_shadow_prefill = dense_shadow_gb(turbo_cache)
            turbo_index_shadow_prefill = index_shadow_gb(turbo_cache)
            turbo_prefix_shadow_prefill = prefix_shadow_gb(turbo_cache)
            turbo_rotated_block_prefill = rotated_block_cache_gb(turbo_cache)
            turbo_key_cosine = mean_stat(turbo_cache, "key_mean_cosine")
            turbo_value_cosine = mean_stat(turbo_cache, "value_mean_cosine")
            turbo_key_mse = mean_stat(turbo_cache, "key_mse")
            turbo_value_mse = mean_stat(turbo_cache, "value_mse")
        else:
            turbo_logits, turbo_prefill = run_prefill(model, inputs, turbo_cache)
            turbo_last = turbo_logits[0]
            turbo_breakdown = get_cache_breakdown(turbo_cache)
            turbo_dense_shadow_prefill = dense_shadow_gb(turbo_cache)
            turbo_index_shadow_prefill = index_shadow_gb(turbo_cache)
            turbo_prefix_shadow_prefill = prefix_shadow_gb(turbo_cache)
            turbo_rotated_block_prefill = rotated_block_cache_gb(turbo_cache)
            turbo_key_cosine = mean_stat(turbo_cache, "key_mean_cosine")
            turbo_value_cosine = mean_stat(turbo_cache, "value_mean_cosine")
            turbo_key_mse = mean_stat(turbo_cache, "key_mse")
            turbo_value_mse = mean_stat(turbo_cache, "value_mse")

        if args.max_tokens > 0 and not args.throughput_only:
            del turbo_cache
            gc.collect()
            mx.clear_cache()
            turbo_cache = build_qwen_cache(
                model,
                implementation=args.implementation,
                bits=args.bits,
                seed=args.seed,
                block_size=args.block_size,
                recent_window=args.recent_window,
                recent_slack=args.recent_slack,
                dense_shadow=args.dense_shadow,
                compute_stats=True,
                share_quantizers=args.share_quantizers,
                lean_direct=args.lean_direct,
                metal_max_query_length=args.metal_max_query_length,
                adaptive_tail_layers=args.adaptive_tail_layers,
                adaptive_tail_quant_bits=args.adaptive_tail_quant_bits,
                adaptive_tail_group_size=args.adaptive_tail_group_size,
            )
            turbo_generation, turbo_generation_stats = run_generation(
                model,
                processor,
                prompt=templated_prompt,
                image=image_arg,
                cache=turbo_cache,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                prefill_step_size=args.prefill_step_size,
            )

    if baseline_last is not None and turbo_last is not None:
        baseline_probs = stable_softmax(baseline_last)
        turbo_probs = stable_softmax(turbo_last)
    else:
        baseline_probs = None
        turbo_probs = None

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"image={image_arg}")
    print(f"prompt={args.prompt}")
    print(f"repeat={args.repeat}")
    print(f"templated_prompt_chars={len(templated_prompt)}")
    print(f"prompt_tokens={int(inputs['input_ids'].shape[1])}")
    print(f"bits={args.bits}")
    print(f"implementation={args.implementation}")
    print(f"run_mode={args.run_mode}")
    print(f"baseline_cache_mode={args.baseline_cache_mode}")
    print(f"baseline_quant_bits={args.baseline_quant_bits}")
    print(f"baseline_quant_group_size={args.baseline_quant_group_size}")
    print(f"throughput_only={int(args.throughput_only)}")
    print(f"lean_direct={int(args.lean_direct)}")
    print(f"share_quantizers={int(args.share_quantizers)}")
    print(f"adaptive_tail_layers={args.adaptive_tail_layers}")
    print(f"adaptive_tail_quant_bits={args.adaptive_tail_quant_bits}")
    print(f"adaptive_tail_group_size={args.adaptive_tail_group_size}")
    print(f"dense_shadow={int(args.dense_shadow)}")
    print(f"block_size={args.block_size}")
    print(f"recent_window={args.recent_window}")
    print(f"recent_slack={args.recent_slack}")
    print(f"metal_max_query_length={args.metal_max_query_length}")
    if baseline_prefill is not None:
        print(f"baseline_prefill_seconds={baseline_prefill['seconds']:.2f}")
        print(f"baseline_prompt_tps={baseline_prefill.get('prompt_tps', 0.0):.3f}")
        print(f"baseline_cache_storage_gb={baseline_prefill['cache_storage_gb']:.6f}")
        print(f"baseline_linear_state_gb={baseline_breakdown.linear_state_gb:.6f}")
        print(f"baseline_full_attention_gb={baseline_breakdown.full_attention_gb:.6f}")
        print(f"baseline_peak_memory_gb={baseline_prefill['peak_memory_gb']:.6f}")
        print(f"baseline_active_memory_gb={baseline_prefill['active_memory_gb']:.6f}")
    if turbo_prefill is not None:
        print(f"turboquant_prefill_seconds={turbo_prefill['seconds']:.2f}")
        print(f"turboquant_prompt_tps={turbo_prefill.get('prompt_tps', 0.0):.3f}")
        print(f"turboquant_cache_storage_gb={turbo_prefill['cache_storage_gb']:.6f}")
        print(f"turboquant_linear_state_gb={turbo_breakdown.linear_state_gb:.6f}")
        print(f"turboquant_full_attention_gb={turbo_breakdown.full_attention_gb:.6f}")
        print(f"turboquant_packed_storage_gb={turbo_breakdown.packed_storage_gb:.6f}")
        print(
            f"turboquant_quantizer_metadata_gb={turbo_breakdown.quantizer_metadata_gb:.6f}"
        )
        print(f"turboquant_dense_shadow_gb={turbo_dense_shadow_prefill:.6f}")
        print(f"turboquant_index_shadow_gb={turbo_index_shadow_prefill:.6f}")
        print(f"turboquant_prefix_shadow_gb={turbo_prefix_shadow_prefill:.6f}")
        print(f"turboquant_rotated_block_cache_gb={turbo_rotated_block_prefill:.6f}")
        print(f"turboquant_peak_memory_gb={turbo_prefill['peak_memory_gb']:.6f}")
        print(f"turboquant_active_memory_gb={turbo_prefill['active_memory_gb']:.6f}")
        print(f"mean_key_cosine={turbo_key_cosine:.4f}")
        print(f"mean_value_cosine={turbo_value_cosine:.4f}")
        print(f"mean_key_mse={turbo_key_mse:.6f}")
        print(f"mean_value_mse={turbo_value_mse:.6f}")

    if baseline_prefill is not None and turbo_prefill is not None:
        print(
            f"storage_compression_ratio={baseline_prefill['cache_storage_gb'] / max(turbo_prefill['cache_storage_gb'], 1e-12):.4f}"
        )
        print(
            f"full_attention_compression_ratio={baseline_breakdown.full_attention_gb / max(turbo_breakdown.full_attention_gb, 1e-12):.4f}"
        )
        print(
            f"packed_storage_only_ratio={baseline_breakdown.full_attention_gb / max(turbo_breakdown.packed_storage_gb, 1e-12):.4f}"
        )
        print(
            f"logit_top1_match={1.0 if int(np.argmax(baseline_last)) == int(np.argmax(turbo_last)) else 0.0:.4f}"
        )
        print(f"logit_top5_overlap={topk_overlap(baseline_last, turbo_last, 5):.4f}")
        print(f"logit_top10_overlap={topk_overlap(baseline_last, turbo_last, 10):.4f}")
        print(f"logit_l2={np.linalg.norm(baseline_last - turbo_last):.4f}")
        print(f"prob_l1={np.sum(np.abs(baseline_probs - turbo_probs)):.4f}")

    if baseline_generation is not None:
        print(f"baseline_generation_seconds={baseline_generation_stats['seconds']:.2f}")
        print(f"baseline_generation_tokens={baseline_generation.generation_tokens}")
        print(f"baseline_generation_tps={baseline_generation_stats.get('generation_tps', 0.0):.3f}")
    if turbo_generation is not None:
        print(f"turboquant_generation_seconds={turbo_generation_stats['seconds']:.2f}")
        print(f"turboquant_generation_tokens={turbo_generation.generation_tokens}")
        print(
            f"turboquant_generation_tps={turbo_generation_stats.get('generation_tps', 0.0):.3f}"
        )

    if baseline_generation is not None and turbo_generation is not None:
        print(
            f"generation_exact_match={1 if baseline_generation.text.strip() == turbo_generation.text.strip() else 0}"
        )
        print("--- baseline_output ---")
        print(baseline_generation.text.strip())
    if turbo_generation is not None:
        print("--- turboquant_output ---")
        print(turbo_generation.text.strip())


if __name__ == "__main__":
    main()
