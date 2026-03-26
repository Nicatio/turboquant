from __future__ import annotations

from typing import Optional

import numpy as np


def _coerce_rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Specify either rng or seed, not both.")
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def normalize_rows(values: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot normalize vectors with near-zero norm.")
    return array / norms


def sample_unit_sphere(
    num_samples: int,
    dimension: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if dimension <= 0:
        raise ValueError("dimension must be positive.")

    generator = _coerce_rng(rng=rng, seed=seed)
    samples = generator.standard_normal((num_samples, dimension))
    return normalize_rows(samples, axis=1)
