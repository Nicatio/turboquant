from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RotationOperator:
    matrix: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            return np.einsum("ij,j->i", self.matrix, array)
        if array.ndim == 2:
            return np.einsum("ij,nj->ni", self.matrix, array)
        raise ValueError("RotationOperator.apply expects a 1D or 2D array.")

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            return np.einsum("ji,j->i", self.matrix, array)
        if array.ndim == 2:
            return np.einsum("ji,nj->ni", self.matrix, array)
        raise ValueError("RotationOperator.inverse expects a 1D or 2D array.")


def generate_random_rotation(
    dimension: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> RotationOperator:
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if rng is not None and seed is not None:
        raise ValueError("Specify either rng or seed, not both.")

    generator = rng if rng is not None else np.random.default_rng(seed)
    gaussian = generator.standard_normal((dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(gaussian)

    phases = np.sign(np.diag(r_matrix))
    phases[phases == 0.0] = 1.0
    q_matrix = q_matrix * phases[np.newaxis, :]
    return RotationOperator(matrix=q_matrix)
