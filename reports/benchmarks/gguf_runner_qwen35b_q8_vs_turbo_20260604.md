# `gguf-runner` Qwen3.5-35B-A3B Q4_K_M Benchmark

Date: 2026-06-04

## Goal

Check whether `gguf-runner` can run the same `Qwen3.5-35B-A3B` family that our MLX benchmarks use, and compare:

- `--kv-cache-mode q8`
- `--kv-cache-mode turbo`

on the same GGUF model.

## Model

- GGUF runtime: `/tmp/gguf-runner`
- Model file: `/private/tmp/gguf_runner_qwen35b_20260604/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Source repo script: [run_gguf_runner_turbo_repro.py](/Users/pome/Turboquant/scripts/run_gguf_runner_turbo_repro.py)

## Prompt

We generated a local text prompt using the Qwen3.5 tokenizer from the MLX model snapshot.

- `2k` prompt file: `/private/tmp/qwen35b_2k_prompt.txt`
- nominal token length: `2048`

## Result

### `2k` prompt, `context-size=4096`, `max_tokens=1`

| KV mode | tok/s | Max RSS (GB) | Peak Memory Footprint (GB) | Elapsed (s) | Output |
|---|---:|---:|---:|---:|---|
| `q8` | `5.951` | `16.888` | `2.284` | `350.05` | `It seems` |
| `turbo` | `5.840` | `17.123` | `2.186` | `356.75` | `It seems` |

### Delta

- throughput: `turbo` is about `-1.9%` vs `q8`
- peak memory footprint: `turbo` is about `-4.3%`
- max RSS: `turbo` is about `+1.4%`

## Interpretation

This confirms that `gguf-runner` can run the `Qwen3.5-35B-A3B` family on this machine with both:

- `q8` KV cache
- `turbo` KV cache

For this short text-only run:

- output matched exactly
- throughput was very close
- `turbo` reduced `peak memory footprint` slightly
- but did not reduce `max RSS`

So at least on this `2k` text-only case, the benefit is present but modest.

## Long-context attempt

We also attempted much longer prompts:

- `32k`
- `64k`
- `128k`

Those runs were too slow to complete within a reasonable interactive turn budget on this machine, so they were stopped before completion.

That means:

- `Qwen3.5-35B-A3B` support in `gguf-runner` is confirmed
- but we do **not** yet have a completed long-context `35B` `q8` vs `turbo` result from `gguf-runner`

## Bottom line

`gguf-runner` can run `Qwen3.5-35B-A3B-Q4_K_M.gguf` here.

On the completed `2k` benchmark:

- `q8`: `5.951 tok/s`, `2.284 GB` peak footprint
- `turbo`: `5.840 tok/s`, `2.186 GB` peak footprint

So the current evidence is:

- same model family: confirmed
- same runtime comparison (`q8` vs `turbo`): confirmed
- strong long-context peak-memory story for `35B`: not yet reproduced in this repo on this machine
