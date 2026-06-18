# Lean Direct 128k Report

Date: 2026-06-01

## Goal

We completed three follow-up tasks for the MLX TurboQuant direct path:

1. Add a `lean direct` mode that disables:
   - `index_shadow`
   - compressed prefix shadow cache
   - rotated block caches
2. Re-measure isolated `128k` peak memory on Gemma 4 and Qwen.
3. Push the direct path one step further onto the fused packed-attention path so `score + softmax + value` can still stay fused without relying on the prefix shadow cache.

## Code Changes

- Lean direct cache controls and fused-path integration:
  - [src/turboquant/kv_cache.py](/Users/pome/Turboquant/src/turboquant/kv_cache.py)
- Gemma 4 long-context benchmark knobs and memory breakdown:
  - [scripts/run_gemma4_long_context_throughput.py](/Users/pome/Turboquant/scripts/run_gemma4_long_context_throughput.py)
- Qwen long-context benchmark knobs and chunked-throughput mode:
  - [scripts/eval_mlx_vlm_turboquant_kv.py](/Users/pome/Turboquant/scripts/eval_mlx_vlm_turboquant_kv.py)
- Direct-cache regression coverage:
  - [tests/test_direct_kv_cache.py](/Users/pome/Turboquant/tests/test_direct_kv_cache.py)

## What Lean Direct Does

`lean_mode=True` in [TurboQuantDirectKVCache](/Users/pome/Turboquant/src/turboquant/kv_cache.py) now implies:

- `use_index_shadow=False`
- `use_compressed_prefix_cache=False`
- `use_rotated_block_cache=False`

In addition, `direct_attention()` now routes block execution through the fused packed-attention kernel even when there is no persistent prefix shadow cache, by unpacking packed indices on demand for the fused block path.

This keeps the persistent auxiliary memory at:

- `dense_shadow_gb = 0`
- `index_shadow_gb = 0`
- `prefix_shadow_gb = 0`
- `rotated_block_cache_gb = 0`

for the new lean-direct benchmarks below.

## Validation

Relevant regression tests passed:

- `27` targeted tests:
  - `tests.test_kv_cache`
  - `tests.test_mlx_quantizer`
  - `tests.test_direct_kv_cache`
  - `tests.test_mlx_attention`

## Benchmark Method

All reported runs were executed in isolated fresh processes to avoid memory carry-over.

### Gemma 4 setup

- Model: `mlx-community/gemma-4-26b-a4b-it-4bit`
- Script: [run_gemma4_long_context_throughput.py](/Users/pome/Turboquant/scripts/run_gemma4_long_context_throughput.py)
- Prompt target: `~128k` (`127,997` actual tokens)
- Decode: `4` generated tokens

### Qwen setup

- Model: `mlx-community/Qwen3.5-35B-A3B-4bit`
- Script: [eval_mlx_vlm_turboquant_kv.py](/Users/pome/Turboquant/scripts/eval_mlx_vlm_turboquant_kv.py)
- Prompt target: `~128k` (`127,996` actual tokens)
- Decode: `1` generated token
- Mode: `--throughput-only`
  - This uses chunked `generate(..., prefill_step_size=256)` instead of a single full-sequence logits prefill, because the old direct full forward path attempted an impractical giant allocation at `128k`.

## Results

### Gemma 4 at 128k

| Mode | Prompt TPS | Gen TPS | Peak Memory (GB) | Active Memory (GB) | Cache Storage (GB) |
|---|---:|---:|---:|---:|---:|
| `baseline` | `289.333` | `20.970` | `19.179` | `17.307` | `2.836398` |
| `shadow` | `157.524` | `1.676` | `19.807` | `15.346` | `0.809924` |
| `direct lean` | `166.798` | `1.342` | `17.954` | `15.346` | `0.809924` |

Key takeaways:

- `direct lean` reduced peak memory below both:
  - baseline: `19.179 -> 17.954 GB`
  - shadow: `19.807 -> 17.954 GB`
- Relative to baseline, Gemma peak memory improved by about `6.4%`.
- Relative to shadow, Gemma peak memory improved by about `9.4%`.
- `shadow` and `direct lean` had the same packed cache storage, but lean direct removed enough runtime auxiliary state to beat shadow on peak memory.

### Qwen 3.5 at 128k

| Mode | Prompt TPS | Gen TPS | Peak Memory (GB) | Active Memory (GB) | Cache Storage (GB) |
|---|---:|---:|---:|---:|---:|
| `baseline` | `287.883` | `738.575` | `25.283856` | `23.085616` | `2.685829` |
| `shadow` | `158.039` | `78.117` | `24.677641` | `21.130655` | `0.599461` |
| `direct lean` | `152.831` | `65.720` | `24.180907` | `21.130638` | `0.599461` |

Notes:

- The Qwen generation TPS numbers are based on `1` generated token, so they are much noisier than the prefill numbers.
- The useful signal here is prompt throughput and peak memory.

Key takeaways:

- `direct lean` again reduced peak memory below both:
  - baseline: `25.284 -> 24.181 GB`
  - shadow: `24.678 -> 24.181 GB`
- Relative to baseline, Qwen peak memory improved by about `4.4%`.
- Relative to shadow, Qwen peak memory improved by about `2.0%`.
- The whole-model gain stays modest because Qwen still has a large non-TurboQuant linear-state slice.

## Auxiliary Memory Breakdown

For the final lean-direct runs, the new persistent helper-memory counters were all zero:

- `turboquant_dense_shadow_gb=0`
- `turboquant_index_shadow_gb=0`
- `turboquant_prefix_shadow_gb=0`
- `turboquant_rotated_block_cache_gb=0`

This confirms the intended effect of the lean mode: the direct path is no longer carrying the biggest persistent auxiliary caches that previously inflated runtime memory.

## Interpretation

### What improved

- We successfully turned `direct` into a more packed-first runtime path.
- Persistent helper caches that previously distorted peak memory are now removable.
- The lean direct path now consistently beats `shadow` on `128k` peak memory for both Gemma and Qwen.

### What did not improve enough yet

- Throughput is still not competitive with baseline:
  - Gemma direct lean prefill is much slower than baseline.
  - Qwen direct lean prefill is also much slower than baseline.
- The fused packed-attention path is better integrated now, but it is still not a `gguf-runner`-style end-to-end packed-native runtime.

## Bottom Line

The new `lean direct` mode is a real step forward:

- it removes the main persistent auxiliary caches,
- it lowers `128k` peak memory below both baseline and shadow,
- and it keeps the direct path on a fused packed-attention route even without the prefix shadow cache.

But the current MLX implementation is still mainly winning on memory cleanliness, not speed.

The next best target after this work is to keep pushing the packed-native execution path so that the direct path can recover prompt throughput without reintroducing large helper caches.

## Follow-up: Experimental Multi-Query Fused Prefill

After the lean-direct memory work, we also extended the fused block attention kernel to support:

- multi-query inputs
- causal masking inside the kernel

This path is now available as an **opt-in** experiment through:

- `metal_max_query_length`

Relevant files:

- [src/turboquant/metal_kernels.py](/Users/pome/Turboquant/src/turboquant/metal_kernels.py)
- [src/turboquant/kv_cache.py](/Users/pome/Turboquant/src/turboquant/kv_cache.py)
- [scripts/run_gemma4_long_context_throughput.py](/Users/pome/Turboquant/scripts/run_gemma4_long_context_throughput.py)
- [scripts/eval_mlx_vlm_turboquant_kv.py](/Users/pome/Turboquant/scripts/eval_mlx_vlm_turboquant_kv.py)

### What happened

Short-context Gemma benefited slightly:

| Gemma 4 direct lean @ ~2k | Prompt TPS |
|---|---:|
| `metal_max_query_length=1` | `596.307` |
| `metal_max_query_length=256` | `614.939` |

That is a small improvement of about `3.1%`.

However, at long context the same idea did not scale well:

| Experiment | Result |
|---|---|
| Gemma 4 direct lean @ `128k`, old path | `166.798 tok/s` |
| Gemma 4 direct lean @ `128k`, fused multi-query path | `160.819 tok/s` |
| Qwen 3.5 direct lean @ `128k`, fused multi-query path | Severe regression; run was aborted after remaining far slower than the previous `~837.85s` lean-direct result |

### Decision

We kept the multi-query fused path in the repo, but **not enabled by default**.

The safe default remains:

- `metal_max_query_length=1`

This gives us:

- the new kernel for controlled experiments
- no regression in the default direct path

### Interpretation

The experimental fused prefill path helps a little at short context, but it is still not efficient enough for large-context Gemma/Qwen prefill.

So the current best state is:

- `lean direct` for cleaner peak memory
- fused multi-query prefill as an explicit experiment only
- default behavior kept conservative until a future packed-native kernel path wins at long context too
