#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


TOKS_RE = re.compile(r"achieved tok/s:\s*([0-9]+(?:\.[0-9]+)?)")
RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")
FOOTPRINT_RE = re.compile(r"(\d+)\s+peak memory footprint")
ELAPSED_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+real")


def run_once(args: argparse.Namespace) -> dict[str, object]:
    prompt = args.prompt
    if prompt is None and args.prompt_file is not None:
        prompt = args.prompt_file.read_text()
    if prompt is None:
        raise ValueError("Either --prompt or --prompt-file must be provided")

    cmd = [
        "/usr/bin/time",
        "-l",
        str(args.binary),
        "--model",
        str(args.model),
        "--prompt",
        prompt,
        "--debug",
        "--show-tokens",
        "--temperature",
        "0",
        "--kv-cache-mode",
        args.kv_cache_mode,
    ]
    if args.image is not None:
        cmd.extend(["--image", str(args.image)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.context_size is not None:
        cmd.extend(["--context-size", str(args.context_size)])
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.think is not None:
        cmd.extend(["--think", args.think])

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr

    toks = None
    rss_bytes = None
    peak_footprint_bytes = None
    elapsed_s = None

    toks_matches = TOKS_RE.findall(combined)
    if toks_matches:
        toks = float(toks_matches[-1])
    m = RSS_RE.search(combined)
    if m:
        rss_bytes = int(m.group(1))
    m = FOOTPRINT_RE.search(combined)
    if m:
        peak_footprint_bytes = int(m.group(1))
    m = ELAPSED_RE.search(combined)
    if m:
        elapsed_s = float(m.group(1))

    return {
        "command": " ".join(shlex.quote(part) for part in cmd),
        "exit_code": completed.returncode,
        "kv_cache_mode": args.kv_cache_mode,
        "tok_per_s": toks,
        "max_rss_bytes": rss_bytes,
        "max_rss_gb": (rss_bytes / (1024 ** 3)) if rss_bytes is not None else None,
        "peak_memory_footprint_bytes": peak_footprint_bytes,
        "peak_memory_footprint_gb": (
            peak_footprint_bytes / (1024 ** 3)
            if peak_footprint_bytes is not None
            else None
        ),
        "elapsed_s": elapsed_s,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated gguf-runner Q8/Turbo benchmark and parse throughput + peak RSS."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--kv-cache-mode", choices=("q8", "turbo"), required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--think", choices=("yes", "no", "hidden"), default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_once(args)
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"kv_cache_mode={result['kv_cache_mode']}")
        print(f"exit_code={result['exit_code']}")
        print(f"tok_per_s={result['tok_per_s']}")
        print(f"max_rss_gb={result['max_rss_gb']}")
        print(f"peak_memory_footprint_gb={result['peak_memory_footprint_gb']}")
        print(f"elapsed_s={result['elapsed_s']}")
        print(f"command={result['command']}")
    return 0 if result["exit_code"] == 0 else int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
