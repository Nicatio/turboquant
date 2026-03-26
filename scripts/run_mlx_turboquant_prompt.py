from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from turboquant.hf_cache import resolve_cached_model_path
from turboquant.mse_quantizer import TurboQuantMSE


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    topk_a = set(np.argsort(-a)[:k].tolist())
    topk_b = set(np.argsort(-b)[:k].tolist())
    return len(topk_a & topk_b) / float(k)


def bytes_to_gb(value: int) -> float:
    return float(value) / 1e9


def quantize_prompt_embeddings(model, input_ids: mx.array, bits: int, seed: int) -> Tuple[mx.array, Dict[str, float]]:
    base_embeddings = np.asarray(model.model.embed_tokens(input_ids), dtype=np.float64)[0]
    norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    unit_embeddings = base_embeddings / safe_norms
    unit_embeddings = unit_embeddings / np.maximum(
        np.linalg.norm(unit_embeddings, axis=1, keepdims=True),
        1e-12,
    )

    quantizer = TurboQuantMSE(
        dimension=unit_embeddings.shape[1],
        bits=bits,
        seed=seed,
        num_grid_points=8193,
        max_iter=96,
        require_unit_norm=True,
    )
    quantized_unit = quantizer.reconstruct(unit_embeddings)
    quantized_embeddings = quantized_unit * safe_norms

    embedding_cosines = np.sum(unit_embeddings * quantized_unit, axis=1)
    metrics = {
        "mean_embedding_cosine": float(np.mean(embedding_cosines)),
        "min_embedding_cosine": float(np.min(embedding_cosines)),
        "prompt_embedding_mse": float(np.mean(np.sum((base_embeddings - quantized_embeddings) ** 2, axis=1))),
    }
    return mx.array(quantized_embeddings), metrics


def last_token_metrics(model, input_ids: mx.array, quantized_embeddings: mx.array) -> Dict[str, float]:
    base_logits = np.asarray(model(input_ids), dtype=np.float64)[0, -1, :]
    quant_logits = np.asarray(
        model(input_ids, input_embeddings=quantized_embeddings[None, :, :]),
        dtype=np.float64,
    )[0, -1, :]

    base_probs = stable_softmax(base_logits)
    quant_probs = stable_softmax(quant_logits)

    return {
        "top1_match": 1.0 if int(np.argmax(base_logits)) == int(np.argmax(quant_logits)) else 0.0,
        "top5_overlap": topk_overlap(base_logits, quant_logits, 5),
        "top10_overlap": topk_overlap(base_logits, quant_logits, 10),
        "logit_l2": float(np.linalg.norm(base_logits - quant_logits)),
        "prob_l1": float(np.sum(np.abs(base_probs - quant_probs))),
    }


def generate_text(model, tokenizer, prompt_tokens, quantized_embeddings, max_tokens: int, temp: float, seed: int) -> Tuple[str, Dict[str, float]]:
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(seed)
    sampler = make_sampler(temp=temp)
    started = time.time()
    text = generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        input_embeddings=quantized_embeddings,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    elapsed = time.time() - started
    memory_stats = {
        "seconds": elapsed,
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
    }
    return text, memory_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a baseline MLX prompt and a TurboQuant-compressed prompt side by side."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="HF/MLX model repo.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text to run.")
    parser.add_argument("--bits", type=int, required=True, help="TurboQuant bit-width for prompt embeddings.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(resolved_model)
    model_memory = {
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
    }
    prompt_tokens = tokenizer.encode(args.prompt)
    input_ids = mx.array([prompt_tokens])

    quantized_embeddings, embedding_metrics = quantize_prompt_embeddings(
        model=model,
        input_ids=input_ids,
        bits=args.bits,
        seed=args.seed,
    )
    accuracy_metrics = last_token_metrics(model, input_ids, quantized_embeddings)

    baseline_text, baseline_stats = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        quantized_embeddings=None,
        max_tokens=args.max_tokens,
        temp=args.temp,
        seed=args.seed,
    )
    quantized_text, quantized_stats = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        quantized_embeddings=quantized_embeddings,
        max_tokens=args.max_tokens,
        temp=args.temp,
        seed=args.seed,
    )

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"bits={args.bits}")
    print(f"prompt_tokens={len(prompt_tokens)}")
    print(f"top1_match={accuracy_metrics['top1_match']:.4f}")
    print(f"top5_overlap={accuracy_metrics['top5_overlap']:.4f}")
    print(f"top10_overlap={accuracy_metrics['top10_overlap']:.4f}")
    print(f"logit_l2={accuracy_metrics['logit_l2']:.4f}")
    print(f"prob_l1={accuracy_metrics['prob_l1']:.4f}")
    print(f"mean_embedding_cosine={embedding_metrics['mean_embedding_cosine']:.4f}")
    print(f"min_embedding_cosine={embedding_metrics['min_embedding_cosine']:.4f}")
    print(f"prompt_embedding_mse={embedding_metrics['prompt_embedding_mse']:.4f}")
    print(f"model_active_memory_gb={model_memory['active_memory_gb']:.3f}")
    print(f"model_cache_memory_gb={model_memory['cache_memory_gb']:.3f}")
    print(f"baseline_generation_seconds={baseline_stats['seconds']:.2f}")
    print(f"baseline_peak_memory_gb={baseline_stats['peak_memory_gb']:.3f}")
    print(f"baseline_active_memory_gb={baseline_stats['active_memory_gb']:.3f}")
    print(f"baseline_cache_memory_gb={baseline_stats['cache_memory_gb']:.3f}")
    print(f"turboquant_generation_seconds={quantized_stats['seconds']:.2f}")
    print(f"turboquant_peak_memory_gb={quantized_stats['peak_memory_gb']:.3f}")
    print(f"turboquant_active_memory_gb={quantized_stats['active_memory_gb']:.3f}")
    print(f"turboquant_cache_memory_gb={quantized_stats['cache_memory_gb']:.3f}")
    print("--- baseline ---")
    print(baseline_text)
    print("--- turboquant ---")
    print(quantized_text)


if __name__ == "__main__":
    main()
