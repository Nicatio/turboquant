from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def sphere_coordinate_pdf(values: np.ndarray, dimension: int) -> np.ndarray:
    if dimension < 2:
        raise ValueError("dimension must be at least 2 for the sphere coordinate density.")

    x = np.asarray(values, dtype=np.float64)
    density = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) < 1.0
    if not np.any(mask):
        return density

    exponent = 0.5 * (dimension - 3)
    log_normalizer = (
        math.lgamma(0.5 * dimension)
        - 0.5 * math.log(math.pi)
        - math.lgamma(0.5 * (dimension - 1))
    )
    base = np.clip(1.0 - np.square(x[mask]), 0.0, None)
    density[mask] = np.exp(log_normalizer) * np.power(base, exponent)
    return density


def make_discrete_pdf_grid(
    dimension: int,
    num_grid_points: int = 16385,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_grid_points < 3:
        raise ValueError("num_grid_points must be at least 3.")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1).")

    grid = np.linspace(-1.0 + epsilon, 1.0 - epsilon, num_grid_points, dtype=np.float64)
    pdf = sphere_coordinate_pdf(grid, dimension)
    total_mass = np.sum(pdf)
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError("Failed to build a valid discrete approximation of the sphere coordinate density.")
    weights = pdf / total_mass
    return grid, weights
