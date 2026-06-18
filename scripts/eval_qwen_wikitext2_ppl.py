from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import load_dataset

from eval_mlx_vlm_turboquant_kv import build_qwen_cache
from turboquant.mlx_attention import enable_turboquant_qwen3_5_attention
from turboquant.mlx_vlm_utils import load_with_slow_processor


DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
DEFAULT_DATASET = "wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"


def append_csv_row(path: str | None, row: dict[str, object]) -> None:
    if not path:
        return
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_text_tokens(processor, *, split: str, text_limit: int | None) -> list[int]:
    dataset = load_dataset(DEFAULT_DATASET, DEFAULT_CONFIG, split=split)
    texts = [str(item["text"]) for item in dataset if str(item["text"]).strip()]
    if text_limit is not None:
        texts = texts[:text_limit]
    joined = "\n\n".join(texts)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return list(tokenizer.encode(joined))


def cross_entropy_sum(logits: np.ndarray, targets: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    rows = np.arange(targets.shape[0], dtype=np.int64)
    return float(-np.sum(log_probs[rows, targets]))


def evaluate_chunks(model, token_ids: list[int], *, chunk_length: int, num_chunks: int, cache_factory):
    total_nll = 0.0
    total_tokens = 0
    processed_chunks = 0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_length
        stop = start + chunk_length + 1
        chunk = token_ids[start:stop]
        if len(chunk) < 2:
            break

        inputs = mx.array([chunk[:-1]])
        targets = np.asarray(chunk[1:], dtype=np.int64)
        cache = cache_factory()

        mx.clear_cache()
        outputs = model(inputs, cache=cache)
        logits = np.asarray(outputs.logits[0].astype(mx.float32), dtype=np.float64)

        total_nll += cross_entropy_sum(logits, targets)
        total_tokens += targets.shape[0]
        processed_chunks += 1

        del cache
        gc.collect()

    if total_tokens == 0:
        raise ValueError("No tokens were evaluated.")
    return {
        "chunks": processed_chunks,
        "tokens": total_tokens,
        "nll": total_nll,
        "ppl": math.exp(total_nll / total_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Wikitext-2 perplexity for baseline vs TurboQuant on Qwen3.5 MLX."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-limit", type=int, default=None, help="Optional number of text rows to keep.")
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--bits", type=float, default=3.0)
    parser.add_argument(
        "--implementation",
        choices=["direct", "shadow"],
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
    parser.add_argument("--adaptive-tail-layers", type=int, default=0)
    parser.add_argument("--adaptive-tail-quant-bits", type=int, default=8)
    parser.add_argument("--adaptive-tail-group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    resolved_model, model, processor = load_with_slow_processor(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    enable_turboquant_qwen3_5_attention(model)

    token_ids = load_text_tokens(
        processor,
        split=args.split,
        text_limit=args.text_limit,
    )

    baseline_factory = model.language_model.make_cache
    turbo_factory = lambda: build_qwen_cache(
        model,
        implementation=args.implementation,
        bits=args.bits,
        seed=args.seed,
        block_size=args.block_size,
        recent_window=args.recent_window,
        recent_slack=args.recent_slack,
        dense_shadow=args.dense_shadow,
        compute_stats=False,
        share_quantizers=args.share_quantizers,
        adaptive_tail_layers=args.adaptive_tail_layers,
        adaptive_tail_quant_bits=args.adaptive_tail_quant_bits,
        adaptive_tail_group_size=args.adaptive_tail_group_size,
    )

    baseline = evaluate_chunks(
        model,
        token_ids,
        chunk_length=args.chunk_length,
        num_chunks=args.num_chunks,
        cache_factory=baseline_factory,
    )
    turbo = evaluate_chunks(
        model,
        token_ids,
        chunk_length=args.chunk_length,
        num_chunks=args.num_chunks,
        cache_factory=turbo_factory,
    )
    ppl_delta = turbo["ppl"] - baseline["ppl"]
    ppl_ratio = turbo["ppl"] / baseline["ppl"]

    append_csv_row(
        args.csv_out,
        {
            "run_label": args.run_label,
            "model": args.model,
            "resolved_model": resolved_model,
            "dataset": DEFAULT_DATASET,
            "dataset_config": DEFAULT_CONFIG,
            "split": args.split,
            "text_limit": args.text_limit if args.text_limit is not None else "",
            "chunk_length": args.chunk_length,
            "num_chunks": args.num_chunks,
            "bits": args.bits,
            "implementation": args.implementation,
            "dense_shadow": int(args.dense_shadow),
            "block_size": args.block_size,
            "recent_window": args.recent_window,
            "recent_slack": args.recent_slack,
            "share_quantizers": int(args.share_quantizers),
            "adaptive_tail_layers": args.adaptive_tail_layers,
            "adaptive_tail_quant_bits": args.adaptive_tail_quant_bits,
            "adaptive_tail_group_size": args.adaptive_tail_group_size,
            "evaluated_chunks": baseline["chunks"],
            "evaluated_tokens": baseline["tokens"],
            "baseline_ppl": f"{baseline['ppl']:.6f}",
            "turboquant_ppl": f"{turbo['ppl']:.6f}",
            "ppl_delta": f"{ppl_delta:.6f}",
            "ppl_ratio": f"{ppl_ratio:.6f}",
        },
    )

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"dataset={DEFAULT_DATASET}")
    print(f"dataset_config={DEFAULT_CONFIG}")
    print(f"split={args.split}")
    print(f"chunk_length={args.chunk_length}")
    print(f"num_chunks={args.num_chunks}")
    print(f"bits={args.bits}")
    print(f"implementation={args.implementation}")
    print(f"dense_shadow={int(args.dense_shadow)}")
    print(f"recent_window={args.recent_window}")
    print(f"recent_slack={args.recent_slack}")
    print(f"share_quantizers={int(args.share_quantizers)}")
    print(f"adaptive_tail_layers={args.adaptive_tail_layers}")
    print(f"adaptive_tail_quant_bits={args.adaptive_tail_quant_bits}")
    print(f"adaptive_tail_group_size={args.adaptive_tail_group_size}")
    print(f"evaluated_chunks={baseline['chunks']}")
    print(f"evaluated_tokens={baseline['tokens']}")
    print(f"baseline_ppl={baseline['ppl']:.6f}")
    print(f"turboquant_ppl={turbo['ppl']:.6f}")
    print(f"ppl_delta={ppl_delta:.6f}")
    print(f"ppl_ratio={ppl_ratio:.6f}")


if __name__ == "__main__":
    main()
