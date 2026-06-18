from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from eval_mlx_vlm_turboquant_kv import (
    DEFAULT_IMAGE,
    DEFAULT_PROMPT,
    build_prompt_text,
    build_qwen_cache,
    get_cache_breakdown,
    prepare_multimodal_inputs,
    run_prefill,
    stable_softmax,
    topk_overlap,
)
from turboquant.mlx_attention import enable_turboquant_qwen3_5_attention
from turboquant.mlx_vlm_utils import load_with_slow_processor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep longer Qwen3.5 multimodal contexts and report TurboQuant KV-cache scaling."
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
        "--repeats",
        nargs="+",
        type=int,
        default=[1, 8, 16, 32, 64],
        help="Prompt repeat counts to evaluate.",
    )
    parser.add_argument("--bits", type=float, default=3.0)
    parser.add_argument(
        "--implementation",
        choices=["direct", "shadow"],
        default="direct",
    )
    parser.add_argument(
        "--share-quantizers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share one TurboQuant rotation/codebook per compatible full-attention head dimension.",
    )
    parser.add_argument("--adaptive-tail-layers", type=int, default=0)
    parser.add_argument("--adaptive-tail-quant-bits", type=int, default=8)
    parser.add_argument("--adaptive-tail-group-size", type=int, default=64)
    parser.add_argument(
        "--recent-window",
        type=int,
        default=0,
        help="Keep this many newest full-attention tokens dense for speed.",
    )
    parser.add_argument(
        "--recent-slack",
        type=int,
        default=0,
        help="Allow this many extra single-token decode steps to stay dense before flushing overflow.",
    )
    parser.add_argument("--dense-shadow", action="store_true")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    image_arg = str(Path(args.image).expanduser()) if "://" not in args.image else args.image

    resolved_model, model, processor = load_with_slow_processor(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    enable_turboquant_qwen3_5_attention(model)

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"image={image_arg}")
    print(f"bits={args.bits}")
    print(f"implementation={args.implementation}")
    print(f"share_quantizers={int(args.share_quantizers)}")
    print(f"adaptive_tail_layers={args.adaptive_tail_layers}")
    print(f"adaptive_tail_quant_bits={args.adaptive_tail_quant_bits}")
    print(f"adaptive_tail_group_size={args.adaptive_tail_group_size}")
    print(f"recent_window={args.recent_window}")
    print(f"recent_slack={args.recent_slack}")
    print(f"dense_shadow={int(args.dense_shadow)}")
    print("repeat prompt_tokens total_ratio full_ratio packed_ratio top1 top5 prob_l1 base_prefill_s turbo_prefill_s")

    for repeat in args.repeats:
        prompt_text = build_prompt_text(args.prompt, repeat)
        _, inputs = prepare_multimodal_inputs(
            model,
            processor,
            prompt_text,
            image_arg,
        )

        baseline_cache = model.language_model.make_cache()
        baseline_logits, baseline_stats = run_prefill(model, inputs, baseline_cache)
        baseline_breakdown = get_cache_breakdown(baseline_cache)
        baseline_last = baseline_logits[0]

        del baseline_cache
        gc.collect()

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
        turbo_logits, turbo_stats = run_prefill(model, inputs, turbo_cache)
        turbo_breakdown = get_cache_breakdown(turbo_cache)
        turbo_last = turbo_logits[0]

        baseline_probs = stable_softmax(baseline_last)
        turbo_probs = stable_softmax(turbo_last)

        print(
            f"{repeat} "
            f"{int(inputs['input_ids'].shape[1])} "
            f"{baseline_breakdown.total_gb / max(turbo_breakdown.total_gb, 1e-12):.4f} "
            f"{baseline_breakdown.full_attention_gb / max(turbo_breakdown.full_attention_gb, 1e-12):.4f} "
            f"{baseline_breakdown.full_attention_gb / max(turbo_breakdown.packed_storage_gb, 1e-12):.4f} "
            f"{1.0 if int(np.argmax(baseline_last)) == int(np.argmax(turbo_last)) else 0.0:.4f} "
            f"{topk_overlap(baseline_last, turbo_last, 5):.4f} "
            f"{np.sum(np.abs(baseline_probs - turbo_probs)):.4f} "
            f"{baseline_stats['seconds']:.2f} "
            f"{turbo_stats['seconds']:.2f}"
        )

        del turbo_cache
        gc.collect()


if __name__ == "__main__":
    main()
