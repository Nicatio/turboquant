from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from mlx_vlm import apply_chat_template, generate

from turboquant.mlx_vlm_utils import load_with_slow_processor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "assets" / "sample_grid.ppm"


def bytes_to_gb(value: int) -> float:
    return float(value) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a local MLX VLM and run a short multimodal generation."
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-35B-A3B-4bit",
        help="HF/MLX VLM model repo or local path.",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Path or URL to an image. Defaults to the tiny local sample image.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in one short sentence.",
        help="Prompt text to pair with the image.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    args = parser.parse_args()

    image_arg = str(Path(args.image).expanduser()) if "://" not in args.image else args.image
    started = time.time()
    resolved_model, model, processor = load_with_slow_processor(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    loaded_at = time.time()

    prompt = apply_chat_template(
        processor,
        model.config,
        args.prompt,
        num_images=1 if image_arg else 0,
    )

    mx.random.seed(args.seed)
    mx.clear_cache()
    mx.reset_peak_memory()
    result = generate(
        model,
        processor,
        prompt=prompt,
        image=image_arg,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=False,
        prefill_step_size=args.prefill_step_size,
    )
    finished_at = time.time()

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"image={image_arg}")
    print(f"templated_prompt_chars={len(prompt)}")
    print(f"load_seconds={loaded_at - started:.2f}")
    print(f"generation_seconds={finished_at - loaded_at:.2f}")
    print(f"prompt_tokens={result.prompt_tokens}")
    print(f"generation_tokens={result.generation_tokens}")
    print(f"prompt_tps={result.prompt_tps:.3f}")
    print(f"generation_tps={result.generation_tps:.3f}")
    print(f"peak_memory_gb={result.peak_memory:.3f}")
    print(f"active_memory_gb={bytes_to_gb(mx.get_active_memory()):.3f}")
    print(f"cache_memory_gb={bytes_to_gb(mx.get_cache_memory()):.3f}")
    print("--- output ---")
    print(result.text)


if __name__ == "__main__":
    main()
