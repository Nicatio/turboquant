from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from turboquant.mse_quantizer import TurboQuantMSE


def _pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint32).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.uint8)
    bit_planes = ((values[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    return np.packbits(bit_planes.reshape(-1), bitorder="little")


def _unpack_indices(packed: np.ndarray, bits: int, shape: Tuple[int, ...]) -> np.ndarray:
    total_values = int(np.prod(shape))
    if total_values == 0:
        return np.zeros(shape, dtype=np.int64)
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")
    unpacked = unpacked[: total_values * bits].reshape(total_values, bits)
    weights = (1 << np.arange(bits, dtype=np.uint32))[None, :]
    values = np.sum(unpacked.astype(np.uint32) * weights, axis=1, dtype=np.uint32)
    return values.reshape(shape).astype(np.int64)


@dataclass
class _EncodedChunk:
    packed_indices: np.ndarray
    norms: np.ndarray
    shape: Tuple[int, ...]
    original_dtype: np.dtype

    @property
    def storage_nbytes(self) -> int:
        return int(self.packed_indices.nbytes + self.norms.nbytes)


class TurboQuantKVCache:
    def __init__(self, bits: int, seed: int = 0, norm_dtype: np.dtype = np.float16):
        if bits < 1:
            raise ValueError("bits must be at least 1.")
        self.turbo_bits = bits
        self.seed = seed
        self.norm_dtype = norm_dtype
        self.offset = 0

        self._key_chunks: List[_EncodedChunk] = []
        self._value_chunks: List[_EncodedChunk] = []
        self._quantizers: Dict[int, TurboQuantMSE] = {}
        self._reconstruction_stats = {
            "key_sq_error_sum": 0.0,
            "value_sq_error_sum": 0.0,
            "key_cosine_sum": 0.0,
            "value_cosine_sum": 0.0,
            "key_vector_count": 0,
            "value_vector_count": 0,
        }

    def _get_quantizer(self, dimension: int) -> TurboQuantMSE:
        quantizer = self._quantizers.get(dimension)
        if quantizer is None:
            quantizer = TurboQuantMSE(
                dimension=dimension,
                bits=self.turbo_bits,
                seed=self.seed + dimension,
                num_grid_points=8193,
                max_iter=96,
                require_unit_norm=True,
            )
            self._quantizers[dimension] = quantizer
        return quantizer

    def _encode_chunk(self, values: np.ndarray) -> Tuple[_EncodedChunk, np.ndarray]:
        original_dtype = values.dtype
        vectors = np.asarray(values, dtype=np.float64)
        dimension = vectors.shape[-1]
        flattened = vectors.reshape(-1, dimension)

        norms = np.linalg.norm(flattened, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        unit_vectors = np.zeros_like(flattened)
        nonzero = norms[:, 0] > 1e-12
        if np.any(nonzero):
            unit_vectors[nonzero] = flattened[nonzero] / safe_norms[nonzero]
            unit_vectors[nonzero] /= np.maximum(
                np.linalg.norm(unit_vectors[nonzero], axis=1, keepdims=True),
                1e-12,
            )

        quantizer = self._get_quantizer(dimension)
        indices = np.zeros(flattened.shape, dtype=np.int64)
        reconstructed_unit = np.zeros_like(flattened)
        if np.any(nonzero):
            quantized_indices = quantizer.quantize_indices(unit_vectors[nonzero])
            indices[nonzero] = quantized_indices.astype(np.int64, copy=False)
            reconstructed_unit[nonzero] = quantizer.dequantize_indices(quantized_indices)

        reconstructed = reconstructed_unit * norms
        encoded = _EncodedChunk(
            packed_indices=_pack_indices(indices, self.turbo_bits),
            norms=norms.astype(self.norm_dtype, copy=False),
            shape=tuple(values.shape),
            original_dtype=original_dtype,
        )
        return encoded, reconstructed.reshape(values.shape)

    def _decode_chunk(self, encoded: _EncodedChunk) -> np.ndarray:
        shape = encoded.shape
        dimension = shape[-1]
        quantizer = self._get_quantizer(dimension)
        indices = _unpack_indices(encoded.packed_indices, self.turbo_bits, shape)
        reconstructed_unit = quantizer.dequantize_indices(indices.reshape(-1, dimension))
        norms = encoded.norms.astype(np.float64, copy=False)
        reconstructed = reconstructed_unit * norms
        return reconstructed.reshape(shape).astype(encoded.original_dtype, copy=False)

    def _append_stats(self, original: np.ndarray, reconstructed: np.ndarray, kind: str) -> None:
        flat_orig = np.asarray(original, dtype=np.float64).reshape(-1, original.shape[-1])
        flat_recon = np.asarray(reconstructed, dtype=np.float64).reshape(-1, reconstructed.shape[-1])
        sq_errors = np.sum((flat_orig - flat_recon) ** 2, axis=1)
        orig_norms = np.linalg.norm(flat_orig, axis=1)
        recon_norms = np.linalg.norm(flat_recon, axis=1)
        denom = np.maximum(orig_norms * recon_norms, 1e-12)
        cosines = np.sum(flat_orig * flat_recon, axis=1) / denom

        self._reconstruction_stats[f"{kind}_sq_error_sum"] += float(np.sum(sq_errors))
        self._reconstruction_stats[f"{kind}_cosine_sum"] += float(np.sum(cosines))
        self._reconstruction_stats[f"{kind}_vector_count"] += int(flat_orig.shape[0])

    def update_and_fetch(self, keys, values):
        key_np = np.asarray(keys)
        value_np = np.asarray(values)

        encoded_keys, reconstructed_keys = self._encode_chunk(key_np)
        encoded_values, reconstructed_values = self._encode_chunk(value_np)

        self._key_chunks.append(encoded_keys)
        self._value_chunks.append(encoded_values)
        self.offset += key_np.shape[2]

        self._append_stats(key_np, reconstructed_keys, "key")
        self._append_stats(value_np, reconstructed_values, "value")

        all_keys = np.concatenate([self._decode_chunk(chunk) for chunk in self._key_chunks], axis=2)
        all_values = np.concatenate([self._decode_chunk(chunk) for chunk in self._value_chunks], axis=2)
        return (
            mx.array(all_keys.astype(key_np.dtype, copy=False)),
            mx.array(all_values.astype(value_np.dtype, copy=False)),
        )

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return len(self._key_chunks) == 0

    @property
    def nbytes(self):
        chunk_bytes = sum(chunk.storage_nbytes for chunk in self._key_chunks + self._value_chunks)
        quantizer_bytes = 0
        for quantizer in self._quantizers.values():
            quantizer_bytes += quantizer.rotation.matrix.nbytes
            quantizer_bytes += quantizer.codebook.centroids.nbytes
            quantizer_bytes += quantizer.codebook.thresholds.nbytes
        return int(chunk_bytes + quantizer_bytes)

    @property
    def stats(self) -> Dict[str, float]:
        key_count = max(self._reconstruction_stats["key_vector_count"], 1)
        value_count = max(self._reconstruction_stats["value_vector_count"], 1)
        return {
            "key_mse": self._reconstruction_stats["key_sq_error_sum"] / key_count,
            "value_mse": self._reconstruction_stats["value_sq_error_sum"] / value_count,
            "key_mean_cosine": self._reconstruction_stats["key_cosine_sum"] / key_count,
            "value_mean_cosine": self._reconstruction_stats["value_cosine_sum"] / value_count,
            "key_vector_count": float(self._reconstruction_stats["key_vector_count"]),
            "value_vector_count": float(self._reconstruction_stats["value_vector_count"]),
        }
