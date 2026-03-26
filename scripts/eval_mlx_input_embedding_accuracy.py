from __future__ import annotations

import argparse
from typing import Dict, List

import mlx.core as mx
import numpy as np
from mlx_lm import load

from turboquant.hf_cache import resolve_cached_model_path
from turboquant.mse_quantizer import TurboQuantMSE


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Vector quantization is useful because",
    "Translate to Korean: machine learning is changing software engineering.",
    "Write one sentence about black holes.",
    "The derivative of x squared is",
    "Summarize why caching helps large language models.",
]


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    topk_a = set(np.argsort(-a)[:k].tolist())
    topk_b = set(np.argsort(-b)[:k].tolist())
    return len(topk_a & topk_b) / float(k)


def evaluate_prompt(
    model,
    tokenizer,
    prompt: str,
    quantizer: TurboQuantMSE,
) -> Dict[str, float]:
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    base_logits = np.asarray(model(input_ids), dtype=np.float64)[0, -1, :]
    base_top1 = int(np.argmax(base_logits))

    base_embeddings = np.asarray(model.model.embed_tokens(input_ids), dtype=np.float64)[0]
    norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    unit_embeddings = base_embeddings / safe_norms
    unit_embeddings = unit_embeddings / np.maximum(
        np.linalg.norm(unit_embeddings, axis=1, keepdims=True),
        1e-12,
    )

    quantized_unit = quantizer.reconstruct(unit_embeddings)
    quantized_embeddings = quantized_unit * safe_norms

    quantized_logits = np.asarray(
        model(input_ids, input_embeddings=mx.array(quantized_embeddings[None, :, :]))
    , dtype=np.float64)[0, -1, :]
    quant_top1 = int(np.argmax(quantized_logits))

    base_probs = stable_softmax(base_logits)
    quant_probs = stable_softmax(quantized_logits)

    embedding_cosines = np.sum(unit_embeddings * quantized_unit, axis=1)

    return {
        "top1_match": 1.0 if base_top1 == quant_top1 else 0.0,
        "top5_overlap": topk_overlap(base_logits, quantized_logits, k=5),
        "top10_overlap": topk_overlap(base_logits, quantized_logits, k=10),
        "logit_l2": float(np.linalg.norm(base_logits - quantized_logits)),
        "prob_l1": float(np.sum(np.abs(base_probs - quant_probs))),
        "mean_embedding_cosine": float(np.mean(embedding_cosines)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TurboQuant on real MLX LLM input embeddings."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="MLX-compatible HF model repo.",
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="TurboQuant bit-widths to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the TurboQuant rotation.",
    )
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(resolved_model)
    dimension = model.args.hidden_size
    prompts: List[str] = DEFAULT_PROMPTS

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"hidden_size={dimension}")
    print(f"num_prompts={len(prompts)}")
    print("bits\ttop1_acc\ttop5_ov\ttop10_ov\tmean_emb_cos\tmean_prob_l1\tmean_logit_l2")

    for bits in args.bits:
        quantizer = TurboQuantMSE(
            dimension=dimension,
            bits=bits,
            seed=args.seed,
            num_grid_points=8193,
            max_iter=96,
            require_unit_norm=True,
        )
        metrics = [evaluate_prompt(model, tokenizer, prompt, quantizer) for prompt in prompts]
        print(
            f"{bits}\t"
            f"{np.mean([m['top1_match'] for m in metrics]):.4f}\t"
            f"{np.mean([m['top5_overlap'] for m in metrics]):.4f}\t"
            f"{np.mean([m['top10_overlap'] for m in metrics]):.4f}\t"
            f"{np.mean([m['mean_embedding_cosine'] for m in metrics]):.4f}\t"
            f"{np.mean([m['prob_l1'] for m in metrics]):.4f}\t"
            f"{np.mean([m['logit_l2'] for m in metrics]):.4f}"
        )


if __name__ == "__main__":
    main()
