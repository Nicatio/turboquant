from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from turboquant.cache import get_codebook
from turboquant.lloyd_max import ScalarCodebook
from turboquant.rotation import RotationOperator, generate_random_rotation


def _uint_dtype(num_levels: int) -> np.dtype:
    if num_levels <= np.iinfo(np.uint8).max:
        return np.uint8
    if num_levels <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def _validate_vectors(values: np.ndarray, dimension: int, require_unit_norm: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError("Input must be a 1D vector or a 2D batch of vectors.")
    if array.shape[-1] != dimension:
        raise ValueError("Input dimension does not match the quantizer dimension.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Input contains non-finite values.")
    if require_unit_norm:
        if array.ndim == 1:
            norms = np.array([np.linalg.norm(array)])
        else:
            norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("TurboQuantMSE currently expects vectors on the unit sphere.")
    return array


@dataclass(frozen=True)
class EncodedVectorMSE:
    indices: np.ndarray


class TurboQuantMSE:
    def __init__(
        self,
        dimension: int,
        bits: int,
        seed: Optional[int] = None,
        rotation: Optional[RotationOperator] = None,
        codebook: Optional[ScalarCodebook] = None,
        num_grid_points: int = 16385,
        max_iter: int = 128,
        tol: float = 1e-10,
        require_unit_norm: bool = True,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be at least 2.")
        if bits < 1:
            raise ValueError("bits must be at least 1.")

        self.dimension = dimension
        self.bits = bits
        self.require_unit_norm = require_unit_norm
        self.rotation = rotation if rotation is not None else generate_random_rotation(dimension, seed=seed)
        self.codebook = codebook if codebook is not None else get_codebook(
            dimension=dimension,
            bits=bits,
            num_grid_points=num_grid_points,
            max_iter=max_iter,
            tol=tol,
        )
        self.index_dtype = _uint_dtype(self.codebook.levels)

    def quantize_indices(self, values: np.ndarray) -> np.ndarray:
        array = _validate_vectors(values, self.dimension, require_unit_norm=self.require_unit_norm)
        rotated = self.rotation.apply(array)
        indices = self.codebook.quantize(rotated).astype(self.index_dtype, copy=False)
        return indices

    def dequantize_indices(self, indices: np.ndarray) -> np.ndarray:
        index_array = np.asarray(indices)
        if index_array.ndim not in (1, 2):
            raise ValueError("Indices must be a 1D array or a 2D batch.")
        if index_array.shape[-1] != self.dimension:
            raise ValueError("Index dimension does not match the quantizer dimension.")
        rotated_reconstruction = self.codebook.dequantize(index_array)
        return self.rotation.inverse(rotated_reconstruction)

    def quantize(self, values: np.ndarray) -> EncodedVectorMSE:
        return EncodedVectorMSE(indices=self.quantize_indices(values))

    def dequantize(self, encoded: EncodedVectorMSE) -> np.ndarray:
        return self.dequantize_indices(encoded.indices)

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        return self.dequantize_indices(self.quantize_indices(values))

