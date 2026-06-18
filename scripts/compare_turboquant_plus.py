from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eval_mlx_vlm_turboquant_kv import (
    build_prompt_text,
    build_qwen_cache,
    get_cache_breakdown,
    prepare_multimodal_inputs,
    run_generation,
    run_prefill,
)
from turboquant.mlx_attention import enable_turboquant_qwen3_5_attention
from turboquant.mlx_vlm_utils import load_with_slow_processor


DEFAULT_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"
DEFAULT_IMAGE = str(Path(__file__).resolve().parents[1] / "assets" / "sample_grid.ppm")
DEFAULT_PROMPT = (
    "What colors appear in the four quadrants of this image? "
    "Answer with only the colors from top-left to bottom-right."
)


PUBLISHED = {
    "model": "Qwen3.5-35B-A3B via llama.cpp Metal on M5 Max 128GB",
    "compression_vs_f16": 4.6,
    "prefill_q8_tok_s": 2694.0,
    "prefill_turbo_tok_s": 2747.0,
    "prefill_ratio": 2747.0 / 2694.0,
    "decode_q8_tok_s": 85.2,
    "decode_turbo_tok_s": 78.4,
    "decode_ratio": 78.4 / 85.2,
    "ppl_q8": 5.414,
    "ppl_turbo": 5.460,
}


def tok_per_second(tokens: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return float(tokens) / seconds


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local TurboQuant numbers against the published turboquant_plus Qwen results."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--repeat", type=int, default=32)
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
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prefill-step-size", type=int, default=256)
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

    prompt_text = build_prompt_text(args.prompt, args.repeat)
    templated_prompt, inputs = prepare_multimodal_inputs(
        model,
        processor,
        prompt_text,
        args.image,
    )

    baseline_cache = model.language_model.make_cache()
    _, baseline_prefill = run_prefill(model, inputs, baseline_cache)
    baseline_breakdown = get_cache_breakdown(baseline_cache)
    baseline_generation, baseline_generation_stats = run_generation(
        model,
        processor,
        prompt=templated_prompt,
        image=args.image,
        cache=model.language_model.make_cache(),
        max_tokens=args.max_tokens,
        temperature=0.0,
        prefill_step_size=args.prefill_step_size,
    )

    turbo_cache = build_qwen_cache(
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
    _, turbo_prefill = run_prefill(model, inputs, turbo_cache)
    turbo_breakdown = get_cache_breakdown(turbo_cache)
    turbo_generation, turbo_generation_stats = run_generation(
        model,
        processor,
        prompt=templated_prompt,
        image=args.image,
        cache=build_qwen_cache(
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
        ),
        max_tokens=args.max_tokens,
        temperature=0.0,
        prefill_step_size=args.prefill_step_size,
    )

    prompt_tokens = int(inputs["input_ids"].shape[1])
    baseline_prefill_tok_s = tok_per_second(prompt_tokens, baseline_prefill["seconds"])
    turbo_prefill_tok_s = tok_per_second(prompt_tokens, turbo_prefill["seconds"])
    baseline_decode_tok_s = tok_per_second(
        baseline_generation.generation_tokens, baseline_generation_stats["seconds"]
    )
    turbo_decode_tok_s = tok_per_second(
        turbo_generation.generation_tokens, turbo_generation_stats["seconds"]
    )
    compression_vs_f16 = baseline_breakdown.total_gb / max(turbo_breakdown.total_gb, 1e-12)
    full_attention_compression = (
        baseline_breakdown.full_attention_gb / max(turbo_breakdown.full_attention_gb, 1e-12)
    )
    prefill_ratio_vs_baseline = turbo_prefill_tok_s / max(baseline_prefill_tok_s, 1e-12)
    decode_ratio_vs_baseline = turbo_decode_tok_s / max(baseline_decode_tok_s, 1e-12)

    append_csv_row(
        args.csv_out,
        {
            "run_label": args.run_label,
            "model": args.model,
            "resolved_model": resolved_model,
            "published_model": PUBLISHED["model"],
            "implementation": args.implementation,
            "dense_shadow": int(args.dense_shadow),
            "bits": args.bits,
            "repeat": args.repeat,
            "prompt_tokens": prompt_tokens,
            "max_tokens": args.max_tokens,
            "recent_window": args.recent_window,
            "recent_slack": args.recent_slack,
            "share_quantizers": int(args.share_quantizers),
            "adaptive_tail_layers": args.adaptive_tail_layers,
            "adaptive_tail_quant_bits": args.adaptive_tail_quant_bits,
            "adaptive_tail_group_size": args.adaptive_tail_group_size,
            "published_compression_vs_f16": f"{PUBLISHED['compression_vs_f16']:.4f}",
            "local_compression_vs_f16": f"{compression_vs_f16:.4f}",
            "published_prefill_tok_s": f"{PUBLISHED['prefill_turbo_tok_s']:.2f}",
            "local_prefill_tok_s": f"{turbo_prefill_tok_s:.2f}",
            "local_baseline_prefill_tok_s": f"{baseline_prefill_tok_s:.2f}",
            "published_prefill_ratio_vs_baseline": f"{PUBLISHED['prefill_ratio']:.4f}",
            "local_prefill_ratio_vs_baseline": f"{prefill_ratio_vs_baseline:.4f}",
            "published_decode_tok_s": f"{PUBLISHED['decode_turbo_tok_s']:.2f}",
            "local_decode_tok_s": f"{turbo_decode_tok_s:.2f}",
            "local_baseline_decode_tok_s": f"{baseline_decode_tok_s:.2f}",
            "published_decode_ratio_vs_baseline": f"{PUBLISHED['decode_ratio']:.4f}",
            "local_decode_ratio_vs_baseline": f"{decode_ratio_vs_baseline:.4f}",
            "local_full_attention_compression": f"{full_attention_compression:.4f}",
            "local_generation_exact_match": int(
                turbo_generation.text.strip() == baseline_generation.text.strip()
            ),
            "published_ppl_q8": f"{PUBLISHED['ppl_q8']:.6f}",
            "published_ppl_turbo": f"{PUBLISHED['ppl_turbo']:.6f}",
        },
    )

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"published_model={PUBLISHED['model']}")
    print(f"implementation={args.implementation}")
    print(f"bits={args.bits}")
    print(f"repeat={args.repeat}")
    print(f"prompt_tokens={prompt_tokens}")
    print("metric\tpublished_turboquant_plus\tlocal_turboquant\tlocal_baseline")
    print(
        f"compression_vs_f16\t{PUBLISHED['compression_vs_f16']:.4f}\t"
        f"{compression_vs_f16:.4f}\t1.0000"
    )
    print(
        f"prefill_tok_s\t{PUBLISHED['prefill_turbo_tok_s']:.2f}\t"
        f"{turbo_prefill_tok_s:.2f}\t{baseline_prefill_tok_s:.2f}"
    )
    print(
        f"prefill_ratio_vs_baseline\t{PUBLISHED['prefill_ratio']:.4f}\t"
        f"{prefill_ratio_vs_baseline:.4f}\t1.0000"
    )
    print(
        f"decode_tok_s\t{PUBLISHED['decode_turbo_tok_s']:.2f}\t"
        f"{turbo_decode_tok_s:.2f}\t{baseline_decode_tok_s:.2f}"
    )
    print(
        f"decode_ratio_vs_baseline\t{PUBLISHED['decode_ratio']:.4f}\t"
        f"{decode_ratio_vs_baseline:.4f}\t1.0000"
    )
    print(
        f"full_attention_compression\t2.0000+\t"
        f"{full_attention_compression:.4f}\t1.0000"
    )
    print(
        f"note\tq8_0 baseline in llama.cpp\tMLX default cache baseline\tMLX default cache baseline"
    )
    print(
        f"published_ppl_q8={PUBLISHED['ppl_q8']:.6f}"
    )
    print(
        f"published_ppl_turbo={PUBLISHED['ppl_turbo']:.6f}"
    )


if __name__ == "__main__":
    main()
