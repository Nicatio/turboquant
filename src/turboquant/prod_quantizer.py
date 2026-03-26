from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from turboquant.mse_quantizer import TurboQuantMSE, _validate_vectors
from turboquant.qjl import QJL


@dataclass(frozen=True)
class EncodedVectorProd:
    mse_indices: Optional[np.ndarray]
    qjl_signs: np.ndarray
    residual_norm: float


class TurboQuantProd:
    def __init__(
        self,
        dimension: int,
        bits: int,
        seed: Optional[int] = None,
        num_grid_points: int = 16385,
        max_iter: int = 128,
        tol: float = 1e-10,
        require_unit_norm: bool = True,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be at least 2.")
        if bits < 1:
            raise ValueError("bits must be at least 1.")

        generator = np.random.default_rng(seed)
        seeds = generator.integers(0, np.iinfo(np.int64).max, size=2, dtype=np.int64)

        self.dimension = dimension
        self.bits = bits
        self.require_unit_norm = require_unit_norm
        self.mse_quantizer = None
        if bits > 1:
            self.mse_quantizer = TurboQuantMSE(
                dimension=dimension,
                bits=bits - 1,
                seed=int(seeds[0]),
                num_grid_points=num_grid_points,
                max_iter=max_iter,
                tol=tol,
                require_unit_norm=require_unit_norm,
            )
        self.qjl = QJL(dimension=dimension, seed=int(seeds[1]))

    def quantize(self, values: np.ndarray) -> EncodedVectorProd:
        vector = _validate_vectors(values, self.dimension, require_unit_norm=self.require_unit_norm)
        if vector.ndim != 1:
            raise ValueError("TurboQuantProd currently expects a single 1D vector.")

        mse_indices = None
        if self.mse_quantizer is None:
            mse_reconstruction = np.zeros(self.dimension, dtype=np.float64)
        else:
            mse_indices = self.mse_quantizer.quantize_indices(vector)
            mse_reconstruction = self.mse_quantizer.dequantize_indices(mse_indices)

        residual = vector - mse_reconstruction
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm == 0.0:
            qjl_signs = np.ones(self.dimension, dtype=np.int8)
        else:
            qjl_signs = self.qjl.quantize_signs(residual)

        return EncodedVectorProd(
            mse_indices=mse_indices,
            qjl_signs=qjl_signs,
            residual_norm=residual_norm,
        )

    def dequantize(self, encoded: EncodedVectorProd) -> np.ndarray:
        if self.mse_quantizer is None or encoded.mse_indices is None:
            mse_reconstruction = np.zeros(self.dimension, dtype=np.float64)
        else:
            mse_reconstruction = self.mse_quantizer.dequantize_indices(encoded.mse_indices)

        if encoded.residual_norm == 0.0:
            correction = np.zeros(self.dimension, dtype=np.float64)
        else:
            correction = encoded.residual_norm * self.qjl.dequantize_signs(encoded.qjl_signs)
        return mse_reconstruction + correction

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        return self.dequantize(self.quantize(values))

    def estimate_inner_product(self, query: np.ndarray, encoded: EncodedVectorProd) -> float:
        query_vector = _validate_vectors(query, self.dimension, require_unit_norm=False)
        if query_vector.ndim != 1:
            raise ValueError("Query must be a single vector.")
        return float(np.dot(query_vector, self.dequantize(encoded)))
