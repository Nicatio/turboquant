# TurboQuant

 TurboQuant implementation for **arXiv:2504.19874**:

- `Q_mse`: random rotation + Lloyd-Max scalar quantization
- `Q_prod`: `Q_mse` plus 1-bit QJL residual correction
- local Apple Silicon MLX experiments for prompt, KV-cache, and multimodal evaluation

This repository is a reproduction focused on correctness, inspectability, and local
experiments on a Mac. It is not a fused production kernel.

## What Is Implemented

- CPU-first NumPy reference implementation of TurboQuant
- scalar codebook construction from the sphere-coordinate distribution
- batched `TurboQuantMSE` encode/decode/reconstruct path
- `TurboQuantProd` inner-product quantizer with residual QJL correction
- synthetic distortion and retrieval evaluation scripts
- MLX-based prompt comparison against a real local LLM
- MLX KV-cache experiment on real key/value tensors
- direct and shadow TurboQuant cache paths for MLX
- multimodal Qwen3.5-35B-A3B evaluation on MLX VLM

## Repository Layout

```text
src/turboquant/
  rotation.py
  distributions.py
  lloyd_max.py
  mse_quantizer.py
  qjl.py
  prod_quantizer.py
  kv_cache.py
  mlx_attention.py
  mlx_vlm_utils.py
  metal_kernels.py
scripts/
  run_synthetic_mse_eval.py
  run_inner_product_eval.py
  run_nn_benchmark.py
  run_mlx_smoke.py
  run_mlx_vlm_smoke.py
  run_mlx_turboquant_prompt.py
  eval_mlx_input_embedding_accuracy.py
  eval_mlx_turboquant_kv_memory.py
  eval_mlx_vlm_turboquant_kv.py
  run_qwen_vlm_turboquant_sweep.py
  run_mlx_needle_benchmark.py
  run_longbench_e_mlx.py
tests/
reports/notes/
```

## Install

Create the environment and install the package:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install mlx-lm
.venv/bin/pip install -e . --no-build-isolation
```

## Core Validation

Run the full test suite:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Current local status:

- `25/25` tests passing

## Synthetic Benchmarks

MSE trend:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_synthetic_mse_eval.py --dimension 64 --samples 4096 --max-bits 4 --seed 0
```

Observed output:

```text
bits    avg_squared_l2_error
1       0.360516
2       0.117362
3       0.035995
4       0.011026
```

Inner-product bias:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_inner_product_eval.py --dimension 64 --bits 3 --trials 512 --seed 0
```

Observed output:

```text
truth=-0.167372
mean_estimate=-0.170570
bias=-0.003198
std=0.050468
```

Retrieval benchmark:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_nn_benchmark.py --dimension 64 --database-size 5000 --query-size 256 --bits 3 --k 10 --seed 0
```

Observed output:

```text
recall@10=0.517969
```

## Local LLM Smoke Test

This uses MLX with a cached local model path when available:

```bash
.venv/bin/python scripts/run_mlx_smoke.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --prompt "What is vector quantization?" \
  --max-tokens 64
```

## Local Multimodal Smoke Test

This uses the official MLX VLM conversion of Qwen3.5-35B-A3B:

```bash
.venv/bin/python scripts/run_mlx_vlm_smoke.py \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --image assets/sample_grid.ppm \
  --prompt "What colors appear in the four quadrants of this image? Answer with only the colors." \
  --max-tokens 96 \
  --prefill-step-size 256
```

Example local result:

```text
peak_memory_gb=20.680
generation_tps=60.981
model correctly identifies the colored quadrants in the image
```

## Prompt-Level TurboQuant Check

Compare baseline generation against TurboQuant-compressed prompt embeddings:

```bash
.venv/bin/python scripts/run_mlx_turboquant_prompt.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bits 3 \
  --prompt "Explain vector quantization in two sentences." \
  --max-tokens 48
```

Example local result at `3` bits:

```text
top1_match=1.0000
top5_overlap=0.8000
top10_overlap=0.9000
mean_embedding_cosine=0.9586
baseline and turboquant output: identical on this prompt
```

To sweep prompt-level accuracy:

```bash
.venv/bin/python scripts/eval_mlx_input_embedding_accuracy.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bits 1 2 3 4 \
  --seed 0
```

Observed output on 6 prompts:

```text
bits  top1_acc  top5_ov  top10_ov  mean_emb_cos  mean_prob_l1  mean_logit_l2
1     0.5000    0.5000   0.5500    0.6375        0.9137        445.6420
2     0.8333    0.8333   0.8833    0.8838        0.2941        134.2337
3     1.0000    0.9333   0.9500    0.9623        0.1081        54.4015
4     1.0000    1.0000   0.9833    0.9858        0.0576        30.0642
```

## KV-Cache Memory Experiment

This is the most relevant local memory experiment in the repo right now. It applies TurboQuant to
the actual key/value tensors stored during model prefill and compares storage and output drift to
the baseline cache.

Run:

```bash
.venv/bin/python scripts/eval_mlx_turboquant_kv_memory.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bits 3 \
  --repeat 64
```

Observed local result:

```text
prompt_tokens=705
baseline_cache_storage_gb=0.088080
turboquant_cache_storage_gb=0.019465
storage_compression_ratio=4.5250
baseline_peak_memory_gb=2.711483
turboquant_peak_memory_gb=2.049048
logit_top1_match=1.0000
logit_top5_overlap=0.8000
logit_top10_overlap=0.8000
prob_l1=0.0155
mean_key_cosine=0.9829
mean_value_cosine=0.9831
```

## Qwen3.5 Multimodal KV-Cache Experiment

This benchmark applies TurboQuant to the full-attention cache in the multimodal
`Qwen3.5-35B-A3B` MLX model and prints both whole-cache and full-attention-only
compression numbers.

Run:

```bash
.venv/bin/python scripts/eval_mlx_vlm_turboquant_kv.py \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --image assets/sample_grid.ppm \
  --bits 3 \
  --implementation direct \
  --repeat 32 \
  --max-tokens 8 \
  --prefill-step-size 256
```

Observed local result:

```text
prompt_tokens=844
baseline_cache_storage_gb=0.053903
turboquant_cache_storage_gb=0.036981
baseline_linear_state_gb=0.032932
baseline_full_attention_gb=0.020972
turboquant_linear_state_gb=0.032932
turboquant_full_attention_gb=0.004049
turboquant_packed_storage_gb=0.003525
turboquant_quantizer_metadata_gb=0.000524
storage_compression_ratio=1.4576
full_attention_compression_ratio=5.1792
packed_storage_only_ratio=5.9497
logit_top1_match=1.0000
prob_l1=0.0011
generation_exact_match=1
```

Important note:

- This benchmark now shares one TurboQuant rotation/codebook across compatible Qwen full-attention layers by default. On this run, that reduced the measured Qwen quantizer metadata from roughly `5.24 MB` to `0.52 MB`.
- The whole-model ratio is still lower than the Llama benchmark because this model uses `full_attention_interval=4`, so only `10/40` layers use standard full-attention KV caches. The other layers use linear-state caches that are not compressed by the current TurboQuant path.
- The `packed_storage_only_ratio` is the best indicator of the asymptotic compression on the compressible slice.

## Qwen Long-Context Sweep

This prefill-only sweep shows how the Qwen multimodal compression ratio changes
as the prompt grows:

```bash
.venv/bin/python scripts/run_qwen_vlm_turboquant_sweep.py \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --image assets/sample_grid.ppm \
  --bits 3 \
  --implementation direct \
  --repeats 1 8 16 32 64 128
```

Observed local result:

```text
repeat prompt_tokens total_ratio full_ratio packed_ratio top1 top5 prob_l1
1      100          1.1270      5.5656     12.5548      1.0000 1.0000 0.0268
8      268          1.2557      6.3794     9.3683       1.0000 0.6000 0.0048
16     460          1.2273      4.2879     5.4584       1.0000 0.6000 0.0019
32     844          1.4576      5.1792     5.9497       1.0000 0.8000 0.0011
64     1612         1.7326      5.0575     5.4515       1.0000 0.6000 0.0006
128    3148         2.1692      4.9854     5.1843       1.0000 0.6000 0.0063
```

Interpretation:

- The whole-model ratio improves with longer prompts because the fixed TurboQuant metadata is amortized while the compressible full-attention cache keeps growing.
- The full-attention slice is now fairly stable around `5x` on this model at longer contexts.
