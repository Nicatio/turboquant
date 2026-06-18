from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval_mlx_vlm_turboquant_kv import (
    DEFAULT_IMAGE,
    DEFAULT_PROMPT,
    build_prompt_text,
    build_qwen_cache,
    get_cache_breakdown,
    prepare_multimodal_inputs,
    run_generation,
    run_prefill,
    stable_softmax,
    topk_overlap,
)
from eval_qwen_wikitext2_ppl import DEFAULT_CONFIG, DEFAULT_DATASET, evaluate_chunks, load_text_tokens
from turboquant.mlx_attention import enable_turboquant_qwen3_5_attention
from turboquant.mlx_vlm_utils import load_with_slow_processor


DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"


@dataclass(frozen=True)
class ModeSpec:
    name: str
    implementation: str
    bits: float
    dense_shadow: bool
    recent_window: int
    recent_slack: int
    adaptive_tail_layers: int
    adaptive_tail_quant_bits: int
    adaptive_tail_group_size: int = 64


MODE_SPECS = {
    "shadow_baseline": ModeSpec(
        name="shadow_baseline",
        implementation="shadow",
        bits=3.0,
        dense_shadow=True,
        recent_window=256,
        recent_slack=8,
        adaptive_tail_layers=0,
        adaptive_tail_quant_bits=8,
    ),
    "direct_baseline": ModeSpec(
        name="direct_baseline",
        implementation="direct",
        bits=3.0,
        dense_shadow=False,
        recent_window=0,
        recent_slack=0,
        adaptive_tail_layers=0,
        adaptive_tail_quant_bits=8,
    ),
    "shadow_adaptive_q8tail": ModeSpec(
        name="shadow_adaptive_q8tail",
        implementation="shadow",
        bits=3.0,
        dense_shadow=True,
        recent_window=256,
        recent_slack=8,
        adaptive_tail_layers=2,
        adaptive_tail_quant_bits=8,
    ),
    "direct_3_5_adaptive_q8tail": ModeSpec(
        name="direct_3_5_adaptive_q8tail",
        implementation="direct",
        bits=3.5,
        dense_shadow=False,
        recent_window=0,
        recent_slack=0,
        adaptive_tail_layers=2,
        adaptive_tail_quant_bits=8,
    ),
}


def tok_per_second(tokens: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(tokens) / seconds


def append_csv_rows(path: str | None, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a structured Qwen TurboQuant mode sweep and summarize speed, compression, and PPL."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODE_SPECS.keys()),
        default=list(MODE_SPECS.keys()),
    )
    parser.add_argument("--csv-out", default="reports/benchmarks/qwen_mode_sweep.csv")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    image_arg = str(Path(args.image).expanduser()) if "://" not in args.image else args.image
    resolved_model, model, processor = load_with_slow_processor(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    enable_turboquant_qwen3_5_attention(model)

    prompt_text = build_prompt_text(args.prompt, args.repeat)
    templated_prompt, inputs = prepare_multimodal_inputs(
        model,
        processor,
        prompt_text,
        image_arg,
    )
    prompt_tokens = int(inputs["input_ids"].shape[1])

    baseline_cache = model.language_model.make_cache()
    baseline_logits, baseline_prefill = run_prefill(model, inputs, baseline_cache)
    baseline_breakdown = get_cache_breakdown(baseline_cache)
    del baseline_cache
    gc.collect()

    baseline_generation, baseline_generation_stats = run_generation(
        model,
        processor,
        prompt=templated_prompt,
        image=image_arg,
        cache=model.language_model.make_cache(),
        max_tokens=args.max_tokens,
        temperature=0.0,
        prefill_step_size=args.prefill_step_size,
    )
    baseline_last = baseline_logits[0]
    baseline_probs = stable_softmax(baseline_last)
    baseline_prefill_tok_s = tok_per_second(prompt_tokens, baseline_prefill["seconds"])
    baseline_decode_tok_s = tok_per_second(
        baseline_generation.generation_tokens,
        baseline_generation_stats["seconds"],
    )

    token_ids = load_text_tokens(processor, split="validation", text_limit=None)
    baseline_ppl = evaluate_chunks(
        model,
        token_ids,
        chunk_length=args.chunk_length,
        num_chunks=args.num_chunks,
        cache_factory=model.language_model.make_cache,
    )

    rows: list[dict[str, object]] = []
    best_row = None
    best_score = None

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"image={image_arg}")
    print(f"repeat={args.repeat}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"chunk_length={args.chunk_length}")
    print(f"num_chunks={args.num_chunks}")
    print("mode prefill_ratio decode_ratio total_ratio full_ratio ppl_ratio exact top1 prob_l1")

    for mode_name in args.modes:
        spec = MODE_SPECS[mode_name]
        cache_kwargs = {
            "implementation": spec.implementation,
            "bits": spec.bits,
            "seed": args.seed,
            "block_size": 256,
            "recent_window": spec.recent_window,
            "recent_slack": spec.recent_slack,
            "dense_shadow": spec.dense_shadow,
            "compute_stats": False,
            "share_quantizers": True,
            "adaptive_tail_layers": spec.adaptive_tail_layers,
            "adaptive_tail_quant_bits": spec.adaptive_tail_quant_bits,
            "adaptive_tail_group_size": spec.adaptive_tail_group_size,
        }

        turbo_cache = build_qwen_cache(model, **cache_kwargs)
        turbo_logits, turbo_prefill = run_prefill(model, inputs, turbo_cache)
        turbo_breakdown = get_cache_breakdown(turbo_cache)
        del turbo_cache
        gc.collect()

        turbo_generation, turbo_generation_stats = run_generation(
            model,
            processor,
            prompt=templated_prompt,
            image=image_arg,
            cache=build_qwen_cache(model, **cache_kwargs),
            max_tokens=args.max_tokens,
            temperature=0.0,
            prefill_step_size=args.prefill_step_size,
        )

        turbo_last = turbo_logits[0]
        turbo_probs = stable_softmax(turbo_last)
        turbo_prefill_tok_s = tok_per_second(prompt_tokens, turbo_prefill["seconds"])
        turbo_decode_tok_s = tok_per_second(
            turbo_generation.generation_tokens,
            turbo_generation_stats["seconds"],
        )

        turbo_ppl = evaluate_chunks(
            model,
            token_ids,
            chunk_length=args.chunk_length,
            num_chunks=args.num_chunks,
            cache_factory=lambda: build_qwen_cache(model, **cache_kwargs),
        )

        row = {
            "mode": spec.name,
            "implementation": spec.implementation,
            "bits": spec.bits,
            "dense_shadow": int(spec.dense_shadow),
            "recent_window": spec.recent_window,
            "recent_slack": spec.recent_slack,
            "adaptive_tail_layers": spec.adaptive_tail_layers,
            "adaptive_tail_quant_bits": spec.adaptive_tail_quant_bits,
            "adaptive_tail_group_size": spec.adaptive_tail_group_size,
            "prompt_tokens": prompt_tokens,
            "max_tokens": args.max_tokens,
            "chunk_length": args.chunk_length,
            "num_chunks": args.num_chunks,
            "baseline_prefill_tok_s": f"{baseline_prefill_tok_s:.2f}",
            "mode_prefill_tok_s": f"{turbo_prefill_tok_s:.2f}",
            "prefill_ratio": f"{turbo_prefill_tok_s / max(baseline_prefill_tok_s, 1e-12):.4f}",
            "baseline_decode_tok_s": f"{baseline_decode_tok_s:.2f}",
            "mode_decode_tok_s": f"{turbo_decode_tok_s:.2f}",
            "decode_ratio": f"{turbo_decode_tok_s / max(baseline_decode_tok_s, 1e-12):.4f}",
            "total_compression_ratio": f"{baseline_breakdown.total_gb / max(turbo_breakdown.total_gb, 1e-12):.4f}",
            "full_attention_compression_ratio": f"{baseline_breakdown.full_attention_gb / max(turbo_breakdown.full_attention_gb, 1e-12):.4f}",
            "packed_storage_only_ratio": f"{baseline_breakdown.full_attention_gb / max(turbo_breakdown.packed_storage_gb, 1e-12):.4f}",
            "baseline_ppl": f"{baseline_ppl['ppl']:.6f}",
            "mode_ppl": f"{turbo_ppl['ppl']:.6f}",
            "ppl_ratio": f"{turbo_ppl['ppl'] / baseline_ppl['ppl']:.6f}",
            "ppl_delta": f"{turbo_ppl['ppl'] - baseline_ppl['ppl']:.6f}",
            "generation_exact_match": int(turbo_generation.text.strip() == baseline_generation.text.strip()),
            "logit_top1_match": float(int(np.argmax(baseline_last)) == int(np.argmax(turbo_last))),
            "logit_top5_overlap": f"{topk_overlap(baseline_last, turbo_last, 5):.4f}",
            "prob_l1": f"{np.sum(np.abs(baseline_probs - turbo_probs)):.4f}",
        }
        rows.append(row)

        prefill_ratio = float(row["prefill_ratio"])
        decode_ratio = float(row["decode_ratio"])
        total_ratio = float(row["total_compression_ratio"])
        ppl_ratio = float(row["ppl_ratio"])
        exact_match = int(row["generation_exact_match"])
        top1_match = float(row["logit_top1_match"])
        # Favor quality first, then decode speed, then compression.
        score = (
            exact_match * 10.0
            + top1_match * 4.0
            + min(prefill_ratio, 1.2) * 1.5
            + min(decode_ratio, 1.2) * 3.0
            + min(total_ratio, 2.0) * 1.0
            + min(1.02 / max(ppl_ratio, 1e-12), 1.1) * 4.0
        )
        if best_score is None or score > best_score:
            best_score = score
            best_row = row

        print(
            f"{spec.name} "
            f"{row['prefill_ratio']} "
            f"{row['decode_ratio']} "
            f"{row['total_compression_ratio']} "
            f"{row['full_attention_compression_ratio']} "
            f"{row['ppl_ratio']} "
            f"{row['generation_exact_match']} "
            f"{row['logit_top1_match']:.4f} "
            f"{row['prob_l1']}"
        )

    append_csv_rows(args.csv_out, rows)

    if best_row is not None:
        print("--- recommended ---")
        print(f"mode={best_row['mode']}")
        print(f"prefill_ratio={best_row['prefill_ratio']}")
        print(f"decode_ratio={best_row['decode_ratio']}")
        print(f"total_compression_ratio={best_row['total_compression_ratio']}")
        print(f"full_attention_compression_ratio={best_row['full_attention_compression_ratio']}")
        print(f"ppl_ratio={best_row['ppl_ratio']}")
        print(f"generation_exact_match={best_row['generation_exact_match']}")


if __name__ == "__main__":
    main()
