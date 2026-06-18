from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from turboquant.metal_kernels import (
    has_turboquant_quantize_kernel,
    turboquant_quantize_rotated,
)
from turboquant.mse_quantizer import TurboQuantMSE


def normalize_bit_width(bits: float | int) -> float:
    value = float(bits)
    if value < 1.0:
        raise ValueError("bits must be at least 1.")
    doubled = round(value * 2.0)
    if abs(value * 2.0 - doubled) > 1e-6:
        raise ValueError("bits must be an integer or half-integer value.")
    return doubled / 2.0


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

    def storage_nbytes(self, shape: tuple[int, ...]) -> int:
        total_values = int(np.prod(shape))
        if total_values == 0:
            return 0
        words = (total_values + self.values_per_word - 1) // self.values_per_word
        return int(words * np.dtype(np.uint32).itemsize)


@dataclass(frozen=True)
class MixedPackedIndexCodec:
    low_bits: int
    high_bits: int
    high_dims: int
    dtype: object = mx.uint32
    _low_codec: PackedIndexCodec = field(init=False, repr=False)
    _high_codec: PackedIndexCodec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.low_bits < 1 or self.high_bits < 1:
            raise ValueError("MixedPackedIndexCodec bit-widths must be positive.")
        object.__setattr__(self, "_low_codec", PackedIndexCodec(self.low_bits, self.dtype))
        object.__setattr__(self, "_high_codec", PackedIndexCodec(self.high_bits, self.dtype))

    def _split_shapes(
        self, shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if self.high_dims > shape[-1]:
            raise ValueError("high_dims exceeds the tensor's last dimension.")
        high_shape = shape[:-1] + (self.high_dims,)
        low_shape = shape[:-1] + (shape[-1] - self.high_dims,)
        return high_shape, low_shape

    def pack(self, indices: mx.array) -> tuple[mx.array, mx.array]:
        shape = tuple(indices.shape)
        high_shape, low_shape = self._split_shapes(shape)
        high = mx.reshape(indices[..., : self.high_dims], high_shape)
        low = mx.reshape(indices[..., self.high_dims :], low_shape)
        return self._high_codec.pack(high), self._low_codec.pack(low)

    def unpack(
        self,
        packed: tuple[mx.array, mx.array],
        shape: tuple[int, ...],
    ) -> mx.array:
        high_shape, low_shape = self._split_shapes(shape)
        packed_high, packed_low = packed
        parts = []
        if high_shape[-1] > 0:
            parts.append(self._high_codec.unpack(packed_high, high_shape))
        if low_shape[-1] > 0:
            parts.append(self._low_codec.unpack(packed_low, low_shape))
        if not parts:
            return mx.zeros(shape, dtype=mx.uint8)
        if len(parts) == 1:
            return parts[0]
        return mx.concatenate(parts, axis=-1)

    def storage_nbytes(self, shape: tuple[int, ...]) -> int:
        high_shape, low_shape = self._split_shapes(shape)
        return self._high_codec.storage_nbytes(high_shape) + self._low_codec.storage_nbytes(low_shape)


def _lookup_from_lut(lut: mx.array, indices: mx.array) -> mx.array:
    levels = lut.shape[1]
    dimension = indices.shape[-1]
    flat_lut = mx.reshape(lut.astype(mx.float32), (-1,))
    offsets = mx.arange(dimension, dtype=mx.uint32) * levels
    offsets = mx.reshape(offsets, (1,) * (indices.ndim - 1) + (dimension,))
    gather_indices = indices.astype(mx.uint32) + offsets
    return mx.take(flat_lut, gather_indices, axis=0)


class MlxTurboQuantMSE:
    def __init__(self, quantizer: TurboQuantMSE, dtype: object = mx.float32) -> None:
        self.dimension = quantizer.dimension
        self.bits = float(quantizer.bits)
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
        self.codec = PackedIndexCodec(bits=int(round(self.bits)))
        self.max_level = int(self.centroids.shape[0])

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
        if has_turboquant_quantize_kernel():
            flat_rotated = mx.reshape(rotated, (-1, self.dimension))
            flat_indices = turboquant_quantize_rotated(flat_rotated, self.thresholds)
            return mx.reshape(flat_indices, rotated.shape)
        comparisons = mx.expand_dims(rotated, axis=-1) > self.thresholds
        return mx.sum(comparisons.astype(mx.uint8), axis=-1).astype(mx.uint8)

    def dequantize_indices(self, indices: mx.array) -> mx.array:
        rotated_reconstruction = self.lookup_centroids(indices)
        return self.unrotate(rotated_reconstruction)

    @property
    def metadata_nbytes(self) -> int:
        return int(
            self.rotation.nbytes
            + self.rotation_transpose.nbytes
            + self.centroids.nbytes
            + self.thresholds.nbytes
        )


class MlxMixedBitTurboQuantMSE:
    def __init__(
        self,
        *,
        dimension: int,
        bits: float,
        seed: int,
        dtype: object = mx.float32,
    ) -> None:
        normalized_bits = normalize_bit_width(bits)
        if float(normalized_bits).is_integer():
            raise ValueError("MlxMixedBitTurboQuantMSE requires a half-integer bit-width.")

        self.dimension = dimension
        self.bits = normalized_bits
        self.dtype = dtype
        self.low_bits = int(np.floor(normalized_bits))
        self.high_bits = self.low_bits + 1
        self.high_dims = int(round((normalized_bits - self.low_bits) * dimension))
        self.high_dims = min(max(self.high_dims, 1), dimension - 1)
        self.low_dims = dimension - self.high_dims

        high_quantizer = TurboQuantMSE(
            dimension=self.high_dims,
            bits=self.high_bits,
            seed=seed,
            num_grid_points=8193,
            max_iter=96,
            require_unit_norm=True,
        )
        low_quantizer = TurboQuantMSE(
            dimension=self.low_dims,
            bits=self.low_bits,
            seed=seed + 500,
            num_grid_points=8193,
            max_iter=96,
            require_unit_norm=True,
        )

        self.high_quantizer = MlxTurboQuantMSE(high_quantizer, dtype=dtype)
        self.low_quantizer = MlxTurboQuantMSE(low_quantizer, dtype=dtype)
        self.codec = MixedPackedIndexCodec(
            low_bits=self.low_bits,
            high_bits=self.high_bits,
            high_dims=self.high_dims,
        )

    def _split(self, values: mx.array) -> tuple[mx.array, mx.array]:
        return values[..., : self.high_dims], values[..., self.high_dims :]

    def rotate(self, values: mx.array) -> mx.array:
        high_values, low_values = self._split(values)
        return mx.concatenate(
            [
                self.high_quantizer.rotate(high_values),
                self.low_quantizer.rotate(low_values),
            ],
            axis=-1,
        )

    def unrotate(self, values: mx.array) -> mx.array:
        high_values, low_values = self._split(values)
        return mx.concatenate(
            [
                self.high_quantizer.unrotate(high_values),
                self.low_quantizer.unrotate(low_values),
            ],
            axis=-1,
        )

    def quantize_indices(self, values: mx.array) -> mx.array:
        high_values, low_values = self._split(values)
        return mx.concatenate(
            [
                self.high_quantizer.quantize_indices(high_values),
                self.low_quantizer.quantize_indices(low_values),
            ],
            axis=-1,
        )

    def lookup_centroids(self, indices: mx.array) -> mx.array:
        high_indices, low_indices = self._split(indices)
        return mx.concatenate(
            [
                self.high_quantizer.lookup_centroids(high_indices),
                self.low_quantizer.lookup_centroids(low_indices),
            ],
            axis=-1,
        )

    def dequantize_indices(self, indices: mx.array) -> mx.array:
        rotated_reconstruction = self.lookup_centroids(indices)
        return self.unrotate(rotated_reconstruction)

    @property
    def metadata_nbytes(self) -> int:
        return self.high_quantizer.metadata_nbytes + self.low_quantizer.metadata_nbytes


def build_mlx_quantizer(
    *,
    dimension: int,
    bits: float | int,
    seed: int,
    dtype: Any = mx.float32,
) -> MlxTurboQuantMSE | MlxMixedBitTurboQuantMSE:
    normalized_bits = normalize_bit_width(bits)
    if float(normalized_bits).is_integer():
        numpy_quantizer = TurboQuantMSE(
            dimension=dimension,
            bits=int(round(normalized_bits)),
            seed=seed,
            num_grid_points=8193,
            max_iter=96,
            require_unit_norm=True,
        )
        return MlxTurboQuantMSE(numpy_quantizer, dtype=dtype)
    return MlxMixedBitTurboQuantMSE(
        dimension=dimension,
        bits=normalized_bits,
        seed=seed,
        dtype=dtype,
    )
