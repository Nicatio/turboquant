from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

from turboquant.distributions import make_discrete_pdf_grid


@dataclass(frozen=True)
class ScalarCodebook:
    centroids: np.ndarray
    thresholds: np.ndarray
    dimension: int
    bits: int
    grid_size: int
    iterations: int
    converged: bool

    @property
    def levels(self) -> int:
        return int(self.centroids.shape[0])

    def quantize(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.thresholds, values, side="right")

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        return self.centroids[np.asarray(indices, dtype=np.int64)]

    def copy(self) -> "ScalarCodebook":
        return ScalarCodebook(
            centroids=self.centroids.copy(),
            thresholds=self.thresholds.copy(),
            dimension=self.dimension,
            bits=self.bits,
            grid_size=self.grid_size,
            iterations=self.iterations,
            converged=self.converged,
        )


def _weighted_quantiles(grid: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.interp(quantiles, cdf, grid)


def _solve_lloyd_max(
    dimension: int,
    bits: int,
    num_grid_points: int,
    max_iter: int,
    tol: float,
) -> ScalarCodebook:
    if dimension < 2:
        raise ValueError("dimension must be at least 2.")
    if bits < 1:
        raise ValueError("bits must be at least 1.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if tol <= 0.0:
        raise ValueError("tol must be positive.")

    num_levels = 1 << bits
    grid, weights = make_discrete_pdf_grid(dimension, num_grid_points=num_grid_points)
    quantiles = (np.arange(num_levels, dtype=np.float64) + 0.5) / num_levels
    centroids = _weighted_quantiles(grid, weights, quantiles)

    converged = False
    for iteration in range(1, max_iter + 1):
        thresholds = 0.5 * (centroids[:-1] + centroids[1:])
        assignments = np.searchsorted(thresholds, grid, side="right")
        updated = centroids.copy()

        left_bounds = np.concatenate((np.array([-1.0]), thresholds))
        right_bounds = np.concatenate((thresholds, np.array([1.0])))

        for index in range(num_levels):
            mask = assignments == index
            bin_weight = float(np.sum(weights[mask]))
            if bin_weight > 0.0:
                updated[index] = float(np.dot(weights[mask], grid[mask]) / bin_weight)
            else:
                updated[index] = 0.5 * (left_bounds[index] + right_bounds[index])

        updated = np.sort(updated)
        delta = float(np.max(np.abs(updated - centroids)))
        centroids = updated
        if delta < tol:
            converged = True
            break

    thresholds = 0.5 * (centroids[:-1] + centroids[1:])
    return ScalarCodebook(
        centroids=centroids,
        thresholds=thresholds,
        dimension=dimension,
        bits=bits,
        grid_size=num_grid_points,
        iterations=iteration,
        converged=converged,
    )


@lru_cache(maxsize=64)
def _cached_lloyd_max(
    dimension: int,
    bits: int,
    num_grid_points: int,
    max_iter: int,
    tol: float,
) -> ScalarCodebook:
    return _solve_lloyd_max(
        dimension=dimension,
        bits=bits,
        num_grid_points=num_grid_points,
        max_iter=max_iter,
        tol=tol,
    )


def build_sphere_codebook(
    dimension: int,
    bits: int,
    num_grid_points: int = 16385,
    max_iter: int = 128,
    tol: float = 1e-10,
) -> ScalarCodebook:
    cached = _cached_lloyd_max(dimension, bits, num_grid_points, max_iter, tol)
    return cached.copy()
