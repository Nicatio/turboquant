from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _validate_vector(values: np.ndarray, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("QJL expects a 1D vector.")
    if array.shape[0] != dimension:
        raise ValueError("Vector dimension does not match the QJL dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Vector contains non-finite values.")
    return array


@dataclass(frozen=True)
class EncodedQJL:
    signs: np.ndarray


class QJL:
    def __init__(
        self,
        dimension: int,
        seed: Optional[int] = None,
        projection: Optional[np.ndarray] = None,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        self.dimension = dimension
        if projection is None:
            generator = np.random.default_rng(seed)
            self.projection = generator.standard_normal((dimension, dimension))
        else:
            matrix = np.asarray(projection, dtype=np.float64)
            if matrix.shape != (dimension, dimension):
                raise ValueError("projection must have shape (dimension, dimension).")
            self.projection = matrix
        self.scale = math.sqrt(math.pi / 2.0) / float(dimension)

    def quantize_signs(self, values: np.ndarray) -> np.ndarray:
        vector = _validate_vector(values, self.dimension)
        projected = np.einsum("ij,j->i", self.projection, vector)
        return np.where(projected >= 0.0, 1, -1).astype(np.int8, copy=False)

    def quantize(self, values: np.ndarray) -> EncodedQJL:
        return EncodedQJL(signs=self.quantize_signs(values))

    def dequantize_signs(self, signs: np.ndarray) -> np.ndarray:
        sign_array = np.asarray(signs, dtype=np.float64)
        if sign_array.ndim != 1 or sign_array.shape[0] != self.dimension:
            raise ValueError("signs must be a 1D array with the QJL dimension.")
        return self.scale * np.einsum("ji,j->i", self.projection, sign_array)

    def dequantize(self, encoded: EncodedQJL) -> np.ndarray:
        return self.dequantize_signs(encoded.signs)
