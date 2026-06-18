# TurboQuant Implementation Plan

## Goal

Implement a local, testable reference reproduction of **TurboQuant** from **arXiv:2504.19874**:

- `Q_mse`: the MSE-oriented quantizer based on random rotation plus per-coordinate scalar quantization
- `Q_prod`: the inner-product-oriented quantizer built from `Q_mse` plus a 1-bit QJL residual correction

The immediate target is a **correct CPU reference implementation** on this Apple Silicon Mac, with tests and small-to-medium benchmarks that validate the paper's core claims locally.

## Paper Anchor

- Title: `TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate`
- arXiv: `2504.19874`
- Key ideas from the paper:
  - Randomly rotate the input vector.
  - Use optimal scalar quantization on rotated coordinates, whose marginal behaves like a Beta-derived distribution on the unit sphere.
  - For unbiased inner-product estimation, quantize the residual using a 1-bit QJL-style correction.

## Local Constraints

- Machine: `macOS 15.6`, `arm64` Apple Silicon
- Available toolchain: `clang`, `cmake`, `git`, `brew`, `node`
- Current Python: system `3.9.6`
- No confirmed CUDA / NVIDIA GPU environment

Implication:

- We should start with a **Python CPU-first reference implementation**.
- We should **not** assume we can reproduce the paper's largest LLM/KV-cache experiments exactly on this machine.
- We can still validate the algorithm well using synthetic data, moderate embedding datasets, and local nearest-neighbor experiments.

## Scope

### In Scope For Phase 1

- Reference implementation of `Q_mse`
- Reference implementation of `Q_prod`
- Encode/decode APIs
- Unit tests and statistical validation tests
- Synthetic benchmarks on unit-sphere vectors
- Small nearest-neighbor retrieval benchmark
- Reproducible scripts for plots/tables generated locally

### In Scope For Phase 2

- Performance cleanup for Apple Silicon CPU
- Caching of codebooks / thresholds
- Optional moderate-scale embedding experiments
- Optional comparison against simple baselines

### Out of Scope Initially

- Full GPU-optimized kernel implementation
- Full LLM serving integration
- Exact reproduction of long-context KV-cache results from the paper
- Exact reproduction of the largest external datasets if they require heavy downloads or proprietary access

## Proposed Tech Stack

### Recommended Environment

- Python `3.12`
- package manager: `uv` if available, otherwise standard `venv`

### Core Dependencies

- `numpy`
- `scipy`
- `pytest`
- `matplotlib`
- `pandas`

### Optional Later

- `hypothesis` for property-based tests
- `numba` for CPU acceleration
- `faiss` only if Apple Silicon support is practical; otherwise use a pure NumPy benchmark path

## Proposed Repository Layout

```text
Turboquant/
  IMPLEMENTATION_PLAN.md
  pyproject.toml
  src/
    turboquant/
      __init__.py
      rotation.py
      distributions.py
      lloyd_max.py
      mse_quantizer.py
      qjl.py
      prod_quantizer.py
      metrics.py
      datasets.py
      nn_eval.py
      cache.py
  tests/
    test_rotation.py
    test_lloyd_max.py
    test_mse_quantizer.py
    test_qjl.py
    test_prod_quantizer.py
    test_distortion_scaling.py
  scripts/
    run_synthetic_mse_eval.py
    run_inner_product_eval.py
    run_nn_benchmark.py
  reports/
    figures/
    notes/
```

## Implementation Strategy

### 1. Paper-to-Code Translation

Before optimizing anything, we should write down a compact design note that maps the paper's math to concrete code objects:

- vector normalization assumptions
- rotation operator construction
- coordinate marginal distribution used for scalar quantizer design
- Lloyd-Max update rule for centroids and thresholds
- bit allocation and codebook indexing
- `Q_prod` residual path and unbiased reconstruction rule

Deliverable:

- short internal note in `reports/notes/` describing the exact formulas we implement

### 2. Rotation Module

Implement the random rotation used by TurboQuant.

Reference path:

- Start with a dense random orthogonal matrix generated from Gaussian matrix + QR decomposition.

Why this first:

- It is the simplest correct reference implementation.
- It is practical for the vector dimensions we can test locally.

Possible later optimization:

- structured fast transforms if we need lower memory or faster throughput

Acceptance criteria:

- rotation preserves norm up to numerical tolerance
- inverse rotation reconstructs original vectors up to floating-point error

### 3. Scalar Quantizer Design (`Q_mse`)

Implement the scalar quantizer used after rotation.

Planned approach:

- model the rotated coordinate distribution for vectors on the unit sphere
- compute Lloyd-Max thresholds and reconstruction values for a given dimension `d` and per-coordinate bit budget
- support lookup-table caching keyed by `(d, bits)`

Important design choice:

- We will implement a **numerically stable reference Lloyd-Max solver** first, even if it is slower than an optimized production version.

Acceptance criteria:

- thresholds are monotone
- centroids lie within corresponding bins
- quantize/dequantize roundtrip is deterministic given a fixed codebook
- distortion decreases as bits increase

### 4. MSE Quantizer Pipeline

Implement the full MSE path:

1. rotate input vector
2. scalar-quantize each coordinate
3. reconstruct rotated vector
4. inverse-rotate back to ambient space

Planned API shape:

- `fit_codebook(d, bits) -> Codebook`
- `quantize_mse(x, codebook, rng=None) -> EncodedVector`
- `dequantize_mse(encoded, codebook) -> np.ndarray`

Acceptance criteria:

- works for batched and single-vector inputs
- reconstruction error is finite and stable
- empirical MSE on unit-sphere samples improves monotonically with bit-width

### 5. Residual QJL Path (`Q_prod`)

Implement the inner-product quantizer described in the paper.

Planned steps:

1. run `Q_mse` to obtain a reconstruction
2. compute residual `r = x - x_hat`
3. apply the 1-bit QJL-style residual sketch
4. combine `x_hat` with the residual estimator during inner-product estimation

Important note:

- This phase needs especially careful reading of the paper formula so that the estimator remains unbiased.
- We should treat unbiasedness as a first-class test target, not just a benchmark statistic.

Acceptance criteria:

- mean bias of estimated inner products is near zero over repeated random trials
- variance decreases as bit budget increases
- results are consistent across different random seeds within expected Monte Carlo error

## 6. Testing Plan

### Unit Tests

- rotation orthogonality and invertibility
- Lloyd-Max solver monotonicity and convergence sanity
- encode/decode shape and dtype checks
- stable behavior for small dimensions and low bit-widths

### Statistical Tests

- empirical MSE vs bit-width on random unit vectors
- empirical inner-product bias/variance for `Q_prod`
- regression tests on fixed random seeds

### Property / Invariant Tests

- norm preservation by rotation
- decoded values remain in valid reconstruction support
- better bit-width should not systematically worsen average distortion

## 7. Benchmark Plan

### Benchmark A: Synthetic Sphere Data

Purpose:

- validate theory with controlled inputs

Method:

- sample vectors uniformly from the unit sphere
- measure `||x - x_hat||_2^2`
- measure inner-product estimation error across random query vectors

Targets:

- reproduce the expected downward distortion trend as bits increase
- compare empirical behavior against the theoretical lower-bound scaling discussed in the paper

### Benchmark B: Moderate Nearest-Neighbor Search

Purpose:

- test whether quantized vectors preserve retrieval quality

Method:

- start with a moderate public embedding dataset or synthetic embedding-like dataset
- compare nearest-neighbor ranking before and after quantization
- report `Recall@k`

Local-first choice:

- begin with a small benchmark that runs comfortably on CPU
- only scale up if downloads and runtime are reasonable on this machine

### Benchmark C: Optional KV-Style Proxy

Purpose:

- approximate the paper's KV-cache motivation without reproducing a full large-model deployment

Method:

- treat transformer-like embeddings as vectors
- compare inner-product or attention-score preservation before and after quantization

Note:

- this is a proxy validation, not a full reproduction of the paper's LLM experiments

## 8. Milestones

### Milestone 0: Environment Bootstrap

- install or select Python `3.12`
- create isolated environment
- add package metadata and test runner

Exit condition:

- `pytest` runs successfully in a new project skeleton

### Milestone 1: Core Math Infrastructure

- rotation module
- distribution helpers
- Lloyd-Max solver

Exit condition:

- codebook generation works for representative `(d, bits)` pairs

### Milestone 2: `Q_mse` End-to-End

- encode/decode pipeline
- synthetic MSE benchmark

Exit condition:

- local script produces a distortion curve over several bit-widths

### Milestone 3: `Q_prod` End-to-End

- residual QJL implementation
- unbiasedness validation

Exit condition:

- local script shows near-zero empirical bias for inner-product estimation

### Milestone 4: Retrieval Benchmark

- nearest-neighbor evaluation harness
- baseline comparison

Exit condition:

- recall metrics are generated and saved locally

### Milestone 5: Refinement

- caching
- performance cleanup
- documentation

Exit condition:

- implementation is reproducible from a clean setup on this machine

## 9. Risks And Unknowns

- The paper's exact scalar quantizer construction may require careful numerical treatment.
- The `Q_prod` residual estimator is easy to implement incorrectly if we shortcut the derivation.
- Exact reproduction of paper-scale experiments may require datasets or hardware we do not have locally.
- Apple Silicon compatibility may affect optional third-party ANN tooling.

Mitigation:

- prioritize correctness and tests over speed
- keep the first version simple and fully inspectable
- validate every stage with synthetic experiments before attempting external benchmarks

## 10. Success Criteria

We will consider the local implementation successful if:

- `Q_mse` and `Q_prod` are both implemented and documented
- tests pass locally on this machine
- synthetic experiments reproduce the expected qualitative trends from the paper
- the project can be re-run from a fresh environment with a short setup sequence

## 11. Immediate Next Step

After approving this plan, the next practical action is:

1. bootstrap a Python project in this workspace
2. install the minimum scientific Python stack
3. implement the rotation and scalar quantizer foundation first

That keeps us on the shortest path to a correct local reproduction.
