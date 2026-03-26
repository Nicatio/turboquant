from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from turboquant.mse_quantizer import TurboQuantMSE


@dataclass(frozen=True)
class PackedIndexCodec:
    bits: int
    dtype: object = mx.uint32

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be at least 1.")
        object.__setattr__(self, "values_per_word", 32 // self.bits)
        if self.values_per_word < 1:
            raise ValueError("bits must be at most 32.")
        weights = [1 << (self.bits * i) for i in range(self.values_per_word)]
        object.__setattr__(self, "weights", mx.array(weights, dtype=self.dtype))
        object.__setattr__(self, "max_level", 1 << self.bits)

    def pack(self, indices: mx.array) -> mx.array:
        flat = mx.reshape(indices.astype(self.dtype), (-1,))
        if flat.shape[0] == 0:
            return mx.zeros((0,), dtype=self.dtype)
        pad = (-flat.shape[0]) % self.values_per_word
        if pad:
            flat = mx.concatenate([flat, mx.zeros((pad,), dtype=self.dtype)], axis=0)
        grouped = mx.reshape(flat, (-1, self.values_per_word))
        return mx.sum(grouped * self.weights[None, :], axis=1).astype(self.dtype)

    def unpack(self, packed: mx.array, shape: tuple[int, ...]) -> mx.array:
        total_values = int(np.prod(shape))
        if total_values == 0:
            return mx.zeros(shape, dtype=mx.uint8)
        expanded = mx.expand_dims(packed.astype(self.dtype), axis=-1)
        unpacked = (expanded // self.weights[None, :]) % self.max_level
        flat = mx.reshape(unpacked.astype(mx.uint8), (-1,))[:total_values]
        return mx.reshape(flat, shape)


class MlxTurboQuantMSE:
    def __init__(self, quantizer: TurboQuantMSE, dtype: object = mx.float32) -> None:
        self.dimension = quantizer.dimension
        self.bits = quantizer.bits
        self.dtype = dtype
        rotation = np.asarray(quantizer.rotation.matrix, dtype=np.float32)
        self.rotation = mx.array(rotation, dtype=dtype)
        self.rotation_transpose = mx.array(rotation.T, dtype=dtype)
        self.centroids = mx.array(
            np.asarray(quantizer.codebook.centroids, dtype=np.float32),
            dtype=dtype,
        )
        self.thresholds = mx.array(
            np.asarray(quantizer.codebook.thresholds, dtype=np.float32),
            dtype=dtype,
        )

    def rotate(self, values: mx.array) -> mx.array:
        original_shape = values.shape
        flattened = mx.reshape(values.astype(self.dtype), (-1, self.dimension))
        rotated = mx.matmul(flattened, self.rotation_transpose)
        return mx.reshape(rotated, original_shape)

    def unrotate(self, values: mx.array) -> mx.array:
        original_shape = values.shape
        flattened = mx.reshape(values.astype(self.dtype), (-1, self.dimension))
        reconstructed = mx.matmul(flattened, self.rotation)
        return mx.reshape(reconstructed, original_shape)

    def lookup_centroids(self, indices: mx.array) -> mx.array:
        return mx.take(self.centroids, indices.astype(mx.uint32), axis=0)

    def quantize_indices(self, values: mx.array) -> mx.array:
        rotated = self.rotate(values)
        if self.thresholds.shape[0] == 0:
            return mx.zeros(rotated.shape, dtype=mx.uint8)
        comparisons = mx.expand_dims(rotated, axis=-1) > self.thresholds
        return mx.sum(comparisons.astype(mx.uint8), axis=-1).astype(mx.uint8)

    def dequantize_indices(self, indices: mx.array) -> mx.array:
        rotated_reconstruction = self.lookup_centroids(indices)
        return self.unrotate(rotated_reconstruction)
