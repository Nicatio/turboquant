# Q8 KV Baseline Alignment Report

Date: 2026-06-03

## Goal

Align our local MLX benchmark baseline more closely with `gguf-runner turbo`, which compares:

- same model
- same weights
- same prompt
- different KV cache modes

Specifically, we added a `q8` KV baseline so our comparisons are no longer limited to:

- `default MLX cache` vs `TurboQuant`

and can now also use:

- `MLX QuantizedKVCache(bits=8)` vs `TurboQuant`

## Code Changes

We added `baseline_cache_mode` support to:

- [run_gemma4_long_context_throughput.py](/Users/pome/Turboquant/scripts/run_gemma4_long_context_throughput.py)
- [eval_mlx_vlm_turboquant_kv.py](/Users/pome/Turboquant/scripts/eval_mlx_vlm_turboquant_kv.py)

New baseline options:

- `default`
- `q8`

For `q8`:

- Gemma 4:
  - full-attention layers use `QuantizedKVCache(bits=8, group_size=64)`
  - sliding-window layers keep `RotatingKVCache`
- Qwen 3.5:
  - full-attention layers use `QuantizedKVCache(bits=8, group_size=64)`
  - linear-state layers keep `ArraysCache`

## Validation

The updated scripts passed `py_compile`.

## Benchmark Conditions

### Gemma 4

- Model: `mlx-community/gemma-4-26b-a4b-it-4bit`
- Prompt tokens: `127,997`
- Decode tokens: `4`
- Script: [run_gemma4_long_context_throughput.py](/Users/pome/Turboquant/scripts/run_gemma4_long_context_throughput.py)

### Qwen 3.5

- Model: `mlx-community/Qwen3.5-35B-A3B-4bit`
- Prompt tokens: `127,996`
- Decode tokens: `1`
- Mode: `--throughput-only`
- Script: [eval_mlx_vlm_turboquant_kv.py](/Users/pome/Turboquant/scripts/eval_mlx_vlm_turboquant_kv.py)

All runs were measured in isolated fresh processes.

## Results

### Gemma 4 at 128k

| Mode | Prompt TPS | Gen TPS | Peak Memory (GB) | Cache Storage (GB) |
|---|---:|---:|---:|---:|
| `default baseline` | `289.333` | `20.970` | `19.179` | `2.836398` |
| `q8 baseline` | `272.807` | `12.118` | `17.767` | `1.602366` |
| `lean direct` | `166.798` | `1.342` | `17.954` | `0.809924` |

### Qwen 3.5 at 128k

| Mode | Prompt TPS | Gen TPS | Peak Memory (GB) | Cache Storage (GB) |
|---|---:|---:|---:|---:|
| `default baseline` | `287.883` | `738.575` | `25.283856` | `2.685829` |
| `q8 baseline` | `273.451` | `602.606` | `23.960447` | `1.456996` |
| `lean direct` | `152.831` | `65.720` | `24.180907` | `0.599461` |

### Reference: `gguf-runner` Reproduction

This is not an apples-to-apples comparison with the MLX runs above.

- Runtime: `gguf-runner`
- Model: `Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`
- Input: image + prompt `please describe the image`
- Metric source: [run_gguf_runner_turbo_repro.py](/Users/pome/Turboquant/scripts/run_gguf_runner_turbo_repro.py)

The purpose of this table is to anchor our local MLX results against a packed-native runtime that uses:

- same weights
- same prompt
- different KV cache modes

| Runtime | Model | Mode | Prompt TPS | Gen TPS | Peak Memory (GB) | Cache Storage (GB) |
|---|---|---|---:|---:|---:|---:|
| `MLX` | `Gemma 4 26B A4B 4bit` | `default baseline` | `289.333` | `20.970` | `19.179` | `2.836398` |
| `MLX` | `Gemma 4 26B A4B 4bit` | `q8 baseline` | `272.807` | `12.118` | `17.767` | `1.602366` |
| `MLX` | `Gemma 4 26B A4B 4bit` | `lean direct` | `166.798` | `1.342` | `17.954` | `0.809924` |
| `MLX` | `Qwen 3.5 35B A3B 4bit` | `default baseline` | `287.883` | `738.575` | `25.283856` | `2.685829` |
| `MLX` | `Qwen 3.5 35B A3B 4bit` | `q8 baseline` | `273.451` | `602.606` | `23.960447` | `1.456996` |
| `MLX` | `Qwen 3.5 35B A3B 4bit` | `lean direct` | `152.831` | `65.720` | `24.180907` | `0.599461` |
| `gguf-runner` | `Qwen3-VL-30B-A3B Q4_K_M` | `q8 baseline` | `-` | `14.157` | `15.24` | `-` |
| `gguf-runner` | `Qwen3-VL-30B-A3B Q4_K_M` | `turbo` | `-` | `13.787` | `7.74` | `-` |

Notes:

- `gguf-runner` reports a single achieved `tok/s` value for generation, so `Prompt TPS` is left blank here.
- `gguf-runner` comparison above uses `peak memory footprint`, which matched the external blog's peak-memory framing more closely than `max RSS`.
- `Cache Storage (GB)` is not currently available from the `gguf-runner` reproduction in the same way our MLX scripts report packed KV storage.

## Interpretation

### What changed versus our previous baseline

The new `q8 baseline` is much closer in spirit to `gguf-runner`'s baseline than our previous `default MLX cache` baseline.

That means our repo can now compare:

- same model
- same runtime family
- different KV cache modes

instead of only comparing against the default MLX cache.

### Key finding

`q8 baseline` is stronger than expected.

For both Gemma and Qwen at `128k`:

- `q8 baseline` reduced peak memory substantially versus `default baseline`
- `q8 baseline` also remained much faster than `lean direct`
- and in these runs, `q8 baseline` slightly beat `lean direct` on peak memory too

### What the `gguf-runner` reference makes clearer

The `gguf-runner` reproduction reinforces that our current MLX `lean direct` path is still not yet a packed-native runtime.

- Our `q8 baseline` is already a strong practical baseline inside MLX.
- Our `lean direct` beats `q8 baseline` on packed KV storage size, but not yet on runtime peak memory.
- `gguf-runner turbo` shows the opposite pattern:
  - very large peak-memory reduction
  - only modest throughput loss versus its own `q8` baseline

So the gap is no longer just "we need more compression." The remaining gap is:

- converting packed-storage gains into real runtime peak-memory wins
- and doing so without the large throughput drop we still see in MLX `lean direct`

#### Gemma

- default baseline peak: `19.179 GB`
- q8 baseline peak: `17.767 GB`
- lean direct peak: `17.954 GB`

So for Gemma:

- `q8 baseline` beat default baseline by about `1.412 GB`
- `q8 baseline` also beat lean direct by about `0.187 GB`

#### Qwen

- default baseline peak: `25.284 GB`
- q8 baseline peak: `23.960 GB`
- lean direct peak: `24.181 GB`

So for Qwen:

- `q8 baseline` beat default baseline by about `1.323 GB`
- `q8 baseline` also beat lean direct by about `0.220 GB`

### What lean direct still wins on

Even though `q8 baseline` now looks very strong on peak memory and speed, `lean direct` still wins clearly on packed cache storage:

- Gemma:
  - `q8 baseline`: `1.602 GB`
  - `lean direct`: `0.810 GB`
- Qwen:
  - `q8 baseline`: `1.457 GB`
  - `lean direct`: `0.599 GB`

So the current state is:

- `q8 baseline` is the strongest practical baseline
- `lean direct` is still the strongest storage compression path
- but `lean direct` has not yet translated its storage advantage into better end-to-end peak memory than `q8 baseline`

## Bottom Line

Adding the `q8 KV baseline` was important, because it changes the comparison story.

Before:

- `lean direct` looked clearly better than the default baseline on peak memory

Now:

- `q8 baseline` is a much tougher and more realistic baseline
- and at `128k`, it currently outperforms `lean direct` on both:
  - peak memory
  - throughput

while `lean direct` still wins only on packed KV storage size.

That means the next optimization target is now clearer:

- not just reducing stored KV size
- but converting that storage advantage into real runtime memory and throughput wins against an `8-bit` KV baseline
