from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Iterable, List

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from turboquant.benchmark_utils import (
    bytes_to_gb,
    cache_nbytes,
    contains_answer,
    encode_chat_prompt,
    exact_match,
    matches_integer,
)
from turboquant.hf_cache import resolve_cached_model_path
from turboquant.kv_cache import TurboQuantKVCache


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "longbench" / "data"
DEFAULT_TASKS = ["passage_count", "passage_retrieval_en"]
DEFAULT_SYSTEM_PROMPT = (
    "You answer long-context benchmark questions. "
    "Return only the shortest correct answer with no extra explanation."
)


def make_turbo_cache(model, bits: int, seed: int):
    return [
        TurboQuantKVCache(bits=bits, seed=seed + i, use_dense_shadow=True)
        for i, _ in enumerate(model.layers)
    ]


def run_generation(model, tokenizer, prompt_tokens, cache, max_tokens: int, seed: int):
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.random.seed(seed)
    started = time.time()
    text = generate(
        model,
        tokenizer,
        prompt=prompt_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0.0),
        max_tokens=max_tokens,
        verbose=False,
    )
    elapsed = time.time() - started
    return text, {
        "seconds": elapsed,
        "peak_memory_gb": bytes_to_gb(mx.get_peak_memory()),
        "active_memory_gb": bytes_to_gb(mx.get_active_memory()),
        "cache_memory_gb": bytes_to_gb(mx.get_cache_memory()),
        "cache_storage_gb": bytes_to_gb(cache_nbytes(cache)),
    }


def coerce_answers(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def is_correct(prediction: str, answers: List[str]) -> bool:
    if matches_integer(prediction, answers):
        return True
    return exact_match(prediction, answers) or contains_answer(prediction, answers)


def config_name(task: str) -> str:
    return task if task.endswith("_e") else f"{task}_e"


def task_file(data_dir: Path, task: str) -> Path:
    return data_dir / f"{config_name(task)}.jsonl"


def build_user_prompt(context: str, question: str) -> str:
    return (
        "Use the long context below to answer the final question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer as briefly as possible."
    )


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description="Run a small LongBench-E adapter on MLX baseline vs TurboQuant KV cache."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing LongBench JSONL files extracted from the official data.zip.",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    resolved_model = resolve_cached_model_path(args.model)
    model, tokenizer = load(resolved_model)

    print(f"model={args.model}")
    print(f"resolved_model={resolved_model}")
    print(f"data_dir={args.data_dir}")
    print(f"bits={args.bits}")

    for task in args.tasks:
        jsonl_path = task_file(args.data_dir, task)
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"LongBench task file not found: {jsonl_path}. Extract the official data.zip first."
            )

        samples = load_dataset("json", data_files=str(jsonl_path), split="train")
        ordered_samples = sorted(samples, key=lambda sample: int(sample["length"]))
        evaluated = 0
        skipped = 0
        baseline_correct = 0
        turbo_correct = 0
        baseline_cache_storage_total = 0.0
        turbo_cache_storage_total = 0.0
        baseline_peak_memory_total = 0.0
        turbo_peak_memory_total = 0.0

        print(f"task={task}")

        for sample in ordered_samples:
            answers = coerce_answers(sample["answers"])
            prompt_tokens = encode_chat_prompt(
                tokenizer,
                user_prompt=build_user_prompt(sample["context"], sample["input"]),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )
            if len(prompt_tokens) > args.max_prompt_tokens:
                skipped += 1
                continue

            baseline_cache = model.make_cache()
            baseline_text, baseline_stats = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                cache=baseline_cache,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            baseline_hit = is_correct(baseline_text, answers)

            del baseline_cache
            gc.collect()
            mx.clear_cache()

            turbo_cache = make_turbo_cache(model, bits=args.bits, seed=args.seed)
            turbo_text, turbo_stats = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                cache=turbo_cache,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            turbo_hit = is_correct(turbo_text, answers)

            evaluated += 1
            baseline_correct += int(baseline_hit)
            turbo_correct += int(turbo_hit)
            baseline_cache_storage_total += baseline_stats["cache_storage_gb"]
            turbo_cache_storage_total += turbo_stats["cache_storage_gb"]
            baseline_peak_memory_total += baseline_stats["peak_memory_gb"]
            turbo_peak_memory_total += turbo_stats["peak_memory_gb"]

            print(
                "sample "
                f"task={task} "
                f"index={evaluated} "
                f"prompt_tokens={len(prompt_tokens)} "
                f"baseline_hit={int(baseline_hit)} "
                f"turbo_hit={int(turbo_hit)} "
                f"baseline_cache_storage_gb={baseline_stats['cache_storage_gb']:.6f} "
                f"turbo_cache_storage_gb={turbo_stats['cache_storage_gb']:.6f} "
                f"baseline_peak_memory_gb={baseline_stats['peak_memory_gb']:.6f} "
                f"turbo_peak_memory_gb={turbo_stats['peak_memory_gb']:.6f}"
            )
            print(f"answers={answers!r}")
            print(f"baseline_answer={baseline_text.strip()!r}")
            print(f"turbo_answer={turbo_text.strip()!r}")

            del turbo_cache
            gc.collect()
            mx.clear_cache()

            if evaluated >= args.max_samples:
                break

        print(f"evaluated={evaluated}")
        print(f"skipped={skipped}")
        print(f"baseline_accuracy={baseline_correct / max(evaluated, 1):.4f}")
        print(f"turbo_accuracy={turbo_correct / max(evaluated, 1):.4f}")
        print(f"baseline_mean_cache_storage_gb={baseline_cache_storage_total / max(evaluated, 1):.6f}")
        print(f"turbo_mean_cache_storage_gb={turbo_cache_storage_total / max(evaluated, 1):.6f}")
        print(f"baseline_mean_peak_memory_gb={baseline_peak_memory_total / max(evaluated, 1):.6f}")
        print(f"turbo_mean_peak_memory_gb={turbo_peak_memory_total / max(evaluated, 1):.6f}")
        print(
            f"mean_storage_compression_ratio="
            f"{baseline_cache_storage_total / max(turbo_cache_storage_total, 1e-12):.4f}"
        )


if __name__ == "__main__":
    main()
