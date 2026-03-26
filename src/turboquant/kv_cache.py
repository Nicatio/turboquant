from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from turboquant.mse_quantizer import TurboQuantMSE
from turboquant.mlx_quantizer import MlxTurboQuantMSE, PackedIndexCodec


@dataclass
class _EncodedChunk:
    packed_indices: mx.array
    norms: mx.array
    shape: Tuple[int, ...]
    original_dtype: object

    @property
    def storage_nbytes(self) -> int:
        return int(self.packed_indices.nbytes + self.norms.nbytes)


class TurboQuantKVCache:
    step = 256

    def __init__(
        self,
        bits: int,
        seed: int = 0,
        norm_dtype: object = mx.float16,
        compute_stats: bool = False,
        use_dense_shadow: bool = False,
    ):
        if bits < 1:
            raise ValueError("bits must be at least 1.")
        self.turbo_bits = bits
        self.seed = seed
        self.norm_dtype = norm_dtype
        self.compute_stats = compute_stats
        self.use_dense_shadow = use_dense_shadow
        self.offset = 0

        self._key_chunks: List[_EncodedChunk] = []
        self._value_chunks: List[_EncodedChunk] = []
        self._quantizers: Dict[int, Tuple[TurboQuantMSE, MlxTurboQuantMSE]] = {}
        self._codec = PackedIndexCodec(bits=bits)
        self._dense_keys = None
        self._dense_values = None
        self._reconstruction_stats = {
            "key_sq_error_sum": 0.0,
            "value_sq_error_sum": 0.0,
            "key_cosine_sum": 0.0,
            "value_cosine_sum": 0.0,
            "key_vector_count": 0,
            "value_vector_count": 0,
        }

    def _reset_chunks(self) -> None:
        self.offset = 0
        self._key_chunks = []
        self._value_chunks = []
        self._dense_keys = None
        self._dense_values = None

    def _decode_all_chunks(self, chunks: List[_EncodedChunk]) -> mx.array:
        if not chunks:
            return mx.zeros((0,), dtype=mx.float32)
        return mx.concatenate([self._decode_chunk(chunk) for chunk in chunks], axis=2)

    def _get_quantizer(self, dimension: int) -> Tuple[TurboQuantMSE, MlxTurboQuantMSE]:
        quantizers = self._quantizers.get(dimension)
        if quantizers is None:
            numpy_quantizer = TurboQuantMSE(
                dimension=dimension,
                bits=self.turbo_bits,
                seed=self.seed + dimension,
                num_grid_points=8193,
                max_iter=96,
                require_unit_norm=True,
            )
            quantizers = (
                numpy_quantizer,
                MlxTurboQuantMSE(numpy_quantizer),
            )
            self._quantizers[dimension] = quantizers
        return quantizers

    def _encode_chunk(self, values: mx.array) -> Tuple[_EncodedChunk, mx.array]:
        original_dtype = values.dtype
        dimension = values.shape[-1]
        flattened = mx.reshape(values.astype(mx.float32), (-1, dimension))

        norms = mx.sqrt(mx.maximum(mx.sum(flattened * flattened, axis=1, keepdims=True), 1e-12))
        safe_norms = mx.maximum(norms, 1e-12)
        unit_vectors = flattened / safe_norms

        _, mlx_quantizer = self._get_quantizer(dimension)
        indices = mlx_quantizer.quantize_indices(unit_vectors)
        reconstructed_unit = mlx_quantizer.dequantize_indices(indices)
        reconstructed = reconstructed_unit * norms
        encoded = _EncodedChunk(
            packed_indices=self._codec.pack(indices),
            norms=norms.astype(self.norm_dtype),
            shape=tuple(values.shape),
            original_dtype=original_dtype,
        )
        return encoded, mx.reshape(reconstructed, values.shape).astype(original_dtype)

    def _decode_chunk(self, encoded: _EncodedChunk) -> mx.array:
        shape = encoded.shape
        dimension = shape[-1]
        _, mlx_quantizer = self._get_quantizer(dimension)
        indices = self._codec.unpack(encoded.packed_indices, shape)
        reconstructed_unit = mlx_quantizer.dequantize_indices(mx.reshape(indices, (-1, dimension)))
        norms = encoded.norms.astype(mx.float32)
        reconstructed = reconstructed_unit * norms
        return mx.reshape(reconstructed, shape).astype(encoded.original_dtype)

    def _append_stats(self, original: mx.array, reconstructed: mx.array, kind: str) -> None:
        if not self.compute_stats:
            return
        flat_orig = mx.reshape(original.astype(mx.float32), (-1, original.shape[-1]))
        flat_recon = mx.reshape(reconstructed.astype(mx.float32), (-1, reconstructed.shape[-1]))
        sq_errors = mx.sum((flat_orig - flat_recon) ** 2, axis=1)
        orig_norms = mx.sqrt(mx.maximum(mx.sum(flat_orig * flat_orig, axis=1), 1e-12))
        recon_norms = mx.sqrt(mx.maximum(mx.sum(flat_recon * flat_recon, axis=1), 1e-12))
        denom = mx.maximum(orig_norms * recon_norms, 1e-12)
        cosines = mx.sum(flat_orig * flat_recon, axis=1) / denom

        self._reconstruction_stats[f"{kind}_sq_error_sum"] += float(mx.sum(sq_errors))
        self._reconstruction_stats[f"{kind}_cosine_sum"] += float(mx.sum(cosines))
        self._reconstruction_stats[f"{kind}_vector_count"] += int(flat_orig.shape[0])

    def _ensure_dense_capacity(self, prev: int, keys: mx.array, values: mx.array) -> None:
        if self._dense_keys is None or (prev + keys.shape[2]) > self._dense_keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self._dense_keys is not None:
                if prev % self.step != 0:
                    self._dense_keys = self._dense_keys[..., :prev, :]
                    self._dense_values = self._dense_values[..., :prev, :]
                self._dense_keys = mx.concatenate([self._dense_keys, new_k], axis=2)
                self._dense_values = mx.concatenate([self._dense_values, new_v], axis=2)
            else:
                self._dense_keys, self._dense_values = new_k, new_v

    def update_and_fetch(self, keys, values):
        encoded_keys, reconstructed_keys = self._encode_chunk(keys)
        encoded_values, reconstructed_values = self._encode_chunk(values)

        self._key_chunks.append(encoded_keys)
        self._value_chunks.append(encoded_values)
        prev = self.offset
        self.offset += keys.shape[2]

        self._append_stats(keys, reconstructed_keys, "key")
        self._append_stats(values, reconstructed_values, "value")

        if self.use_dense_shadow:
            self._ensure_dense_capacity(prev, keys, values)
            self._dense_keys[..., prev : self.offset, :] = reconstructed_keys
            self._dense_values[..., prev : self.offset, :] = reconstructed_values
            return (
                self._dense_keys[..., : self.offset, :],
                self._dense_values[..., : self.offset, :],
            )

        return (
            self._decode_all_chunks(self._key_chunks),
            self._decode_all_chunks(self._value_chunks),
        )

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.offset == 0

    @property
    def state(self):
        if self.use_dense_shadow and self._dense_keys is not None:
            return (
                self._dense_keys[..., : self.offset, :],
                self._dense_values[..., : self.offset, :],
            )
        return (
            self._decode_all_chunks(self._key_chunks),
            self._decode_all_chunks(self._value_chunks),
        )

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self._reset_chunks()
        if keys is None or values is None:
            return

        encoded_keys, reconstructed_keys = self._encode_chunk(keys)
        encoded_values, reconstructed_values = self._encode_chunk(values)
        self._key_chunks.append(encoded_keys)
        self._value_chunks.append(encoded_values)
        self.offset = keys.shape[2]
        if self.use_dense_shadow:
            self._ensure_dense_capacity(0, reconstructed_keys, reconstructed_values)
            self._dense_keys[..., : self.offset, :] = reconstructed_keys
            self._dense_values[..., : self.offset, :] = reconstructed_values

    @property
    def dense_nbytes(self) -> int:
        if self._dense_keys is None:
            return 0
        return int(self._dense_keys.nbytes + self._dense_values.nbytes)

    @property
    def nbytes(self):
        chunk_bytes = sum(chunk.storage_nbytes for chunk in self._key_chunks + self._value_chunks)
        quantizer_bytes = 0
        for numpy_quantizer, _ in self._quantizers.values():
            quantizer_bytes += numpy_quantizer.rotation.matrix.nbytes
            quantizer_bytes += numpy_quantizer.codebook.centroids.nbytes
            quantizer_bytes += numpy_quantizer.codebook.thresholds.nbytes
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
