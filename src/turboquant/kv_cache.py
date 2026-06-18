from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from turboquant.metal_kernels import (
    has_turboquant_fused_attention_kernel,
    has_turboquant_score_kernel,
    has_turboquant_value_kernel,
    turboquant_fused_block_attention,
    turboquant_score_block,
    turboquant_value_block,
)
from turboquant.mlx_quantizer import build_mlx_quantizer, normalize_bit_width


@dataclass(frozen=True)
class _QuantizerSpec:
    dimension: int
    bits: float
    seed: int
    num_grid_points: int = 8193
    max_iter: int = 96


class TurboQuantQuantizerPool:
    def __init__(self) -> None:
        self._quantizers: Dict[_QuantizerSpec, object] = {}

    def get(
        self,
        *,
        dimension: int,
        bits: float,
        seed: int,
        num_grid_points: int = 8193,
        max_iter: int = 96,
    ):
        spec = _QuantizerSpec(
            dimension=dimension,
            bits=normalize_bit_width(bits),
            seed=seed,
            num_grid_points=num_grid_points,
            max_iter=max_iter,
        )
        quantizers = self._quantizers.get(spec)
        if quantizers is None:
            quantizers = build_mlx_quantizer(
                dimension=dimension,
                bits=spec.bits,
                seed=seed,
            )
            self._quantizers[spec] = quantizers
        return quantizers


def quantizer_metadata_nbytes(quantizers) -> int:
    total = 0
    for quantizer in quantizers.values():
        total += int(getattr(quantizer, "metadata_nbytes", 0))
    return int(total)


def _nested_nbytes(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return int(sum(_nested_nbytes(item) for item in value))
    return int(getattr(value, "nbytes", 0))


def _cache_entry_nbytes(cache_entry) -> int:
    try:
        return int(getattr(cache_entry, "nbytes", 0))
    except Exception:
        return int(_nested_nbytes(getattr(cache_entry, "state", None)))


def cache_packed_nbytes(cache_entry) -> int:
    if isinstance(cache_entry, TurboQuantDirectKVCache):
        total = int(sum(block.storage_nbytes for block in cache_entry._blocks))
        if cache_entry._recent_keys is not None:
            total += int(cache_entry._recent_keys.nbytes + cache_entry._recent_values.nbytes)
        return total
    if isinstance(cache_entry, TurboQuantKVCache):
        total = int(
            sum(
                chunk.storage_nbytes
                for chunk in cache_entry._key_chunks + cache_entry._value_chunks
            )
        )
        if cache_entry._recent_keys is not None:
            total += int(cache_entry._recent_keys.nbytes + cache_entry._recent_values.nbytes)
        return total
    return _cache_entry_nbytes(cache_entry)


def cache_quantizer_metadata_nbytes(cache_entry) -> int:
    if isinstance(cache_entry, (TurboQuantDirectKVCache, TurboQuantKVCache)):
        if cache_entry._quantizer_pool is not None:
            return 0
        return quantizer_metadata_nbytes(cache_entry._quantizers)
    return 0


def cache_list_nbytes(cache_entries) -> int:
    total = 0
    seen_pools = set()
    for entry in cache_entries:
        if isinstance(entry, (TurboQuantDirectKVCache, TurboQuantKVCache)):
            total += cache_packed_nbytes(entry)
        else:
            total += _cache_entry_nbytes(entry)
            continue

        pool = entry._quantizer_pool
        if pool is None:
            total += cache_quantizer_metadata_nbytes(entry)
            continue

        pool_id = id(pool)
        if pool_id in seen_pools:
            continue
        seen_pools.add(pool_id)
        total += quantizer_metadata_nbytes(pool._quantizers)
    return int(total)


@dataclass
class _EncodedChunk:
    packed_indices: object | None
    indices: mx.array | None
    packed_nbytes: int
    norms: mx.array
    shape: Tuple[int, ...]
    original_dtype: object

    @property
    def storage_nbytes(self) -> int:
        return int(self.packed_nbytes + self.norms.nbytes)

    @property
    def index_shadow_nbytes(self) -> int:
        if self.indices is None:
            return 0
        return int(self.indices.nbytes)


@dataclass
class _JointEncodedBlock:
    key_packed_indices: object | None
    key_indices: mx.array | None
    key_packed_nbytes: int
    key_norms: mx.array
    key_shape: Tuple[int, ...]
    key_dtype: object
    value_packed_indices: object | None
    value_indices: mx.array | None
    value_packed_nbytes: int
    value_norms: mx.array
    value_shape: Tuple[int, ...]
    value_dtype: object
    start: int
    key_rotated_cache: mx.array | None = None
    value_rotated_cache: mx.array | None = None

    @property
    def length(self) -> int:
        return self.key_shape[2]

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.key_packed_nbytes
            + self.key_norms.nbytes
            + self.value_packed_nbytes
            + self.value_norms.nbytes
        )

    @property
    def rotated_cache_nbytes(self) -> int:
        total = 0
        if self.key_rotated_cache is not None:
            total += int(self.key_rotated_cache.nbytes)
        if self.value_rotated_cache is not None:
            total += int(self.value_rotated_cache.nbytes)
        return total


class TurboQuantKVCache:
    step = 256

    def __init__(
        self,
        bits: float,
        seed: int = 0,
        norm_dtype: object = mx.float16,
        compute_stats: bool = False,
        use_dense_shadow: bool = False,
        recent_window_tokens: int = 0,
        recent_slack_tokens: int = 0,
        quantizer_pool: TurboQuantQuantizerPool | None = None,
    ):
        self.turbo_bits = normalize_bit_width(bits)
        self.seed = seed
        self.norm_dtype = norm_dtype
        self.compute_stats = compute_stats
        self.use_dense_shadow = use_dense_shadow
        self.recent_window_tokens = recent_window_tokens
        self.recent_slack_tokens = recent_slack_tokens
        self._quantizer_pool = quantizer_pool
        self.offset = 0

        self._key_chunks: List[_EncodedChunk] = []
        self._value_chunks: List[_EncodedChunk] = []
        self._quantizers: Dict[int, object] = {}
        self._dense_keys = None
        self._dense_values = None
        self._recent_start = 0
        self._recent_keys = None
        self._recent_values = None
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
        self._recent_start = 0
        self._recent_keys = None
        self._recent_values = None

    def _decode_all_chunks(self, chunks: List[_EncodedChunk]) -> mx.array:
        if not chunks:
            return mx.zeros((0,), dtype=mx.float32)
        return mx.concatenate([self._decode_chunk(chunk) for chunk in chunks], axis=2)

    def _get_quantizer(self, dimension: int):
        if self._quantizer_pool is not None:
            return self._quantizer_pool.get(
                dimension=dimension,
                bits=self.turbo_bits,
                seed=self.seed,
            )

        quantizers = self._quantizers.get(dimension)
        if quantizers is None:
            quantizers = build_mlx_quantizer(
                dimension=dimension,
                bits=self.turbo_bits,
                seed=self.seed + dimension,
            )
            self._quantizers[dimension] = quantizers
        return quantizers

    def _encode_chunk(self, values: mx.array) -> Tuple[_EncodedChunk, mx.array]:
        original_dtype = values.dtype
        dimension = values.shape[-1]
        shape = tuple(values.shape)
        flattened = mx.reshape(values.astype(mx.float32), (-1, dimension))

        norms = mx.sqrt(mx.maximum(mx.sum(flattened * flattened, axis=1, keepdims=True), 1e-12))
        safe_norms = mx.maximum(norms, 1e-12)
        unit_vectors = flattened / safe_norms

        mlx_quantizer = self._get_quantizer(dimension)
        indices = mlx_quantizer.quantize_indices(unit_vectors)
        reconstructed_unit = mlx_quantizer.dequantize_indices(indices)
        reconstructed = reconstructed_unit * norms
        stored_indices = None
        packed_indices = None
        packed_nbytes = mlx_quantizer.codec.storage_nbytes(shape)
        if self.use_dense_shadow:
            stored_indices = mx.reshape(indices, shape)
        else:
            packed_indices = mlx_quantizer.codec.pack(indices)
        encoded = _EncodedChunk(
            packed_indices=packed_indices,
            indices=stored_indices,
            packed_nbytes=packed_nbytes,
            norms=norms.astype(self.norm_dtype),
            shape=shape,
            original_dtype=original_dtype,
        )
        return encoded, mx.reshape(reconstructed, values.shape).astype(original_dtype)

    def _decode_chunk(self, encoded: _EncodedChunk) -> mx.array:
        shape = encoded.shape
        dimension = shape[-1]
        mlx_quantizer = self._get_quantizer(dimension)
        indices = encoded.indices
        if indices is None:
            if encoded.packed_indices is None:
                raise ValueError("Encoded chunk is missing both packed and unpacked indices.")
            indices = mlx_quantizer.codec.unpack(encoded.packed_indices, shape)
        reconstructed_unit = mlx_quantizer.dequantize_indices(mx.reshape(indices, (-1, dimension)))
        norms = encoded.norms.astype(mx.float32)
        reconstructed = reconstructed_unit * norms
        return mx.reshape(reconstructed, shape).astype(encoded.original_dtype)

    def _materialize_chunk_packed_indices(self, chunk: _EncodedChunk) -> mx.array:
        if chunk.packed_indices is None:
            if chunk.indices is None:
                raise ValueError("Encoded chunk is missing both packed and unpacked indices.")
            mlx_quantizer = self._get_quantizer(chunk.shape[-1])
            chunk.packed_indices = mlx_quantizer.codec.pack(chunk.indices)
        return chunk.packed_indices

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

    def _encode_and_store(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        encoded_keys, reconstructed_keys = self._encode_chunk(keys)
        encoded_values, reconstructed_values = self._encode_chunk(values)
        self._key_chunks.append(encoded_keys)
        self._value_chunks.append(encoded_values)
        self._append_stats(keys, reconstructed_keys, "key")
        self._append_stats(values, reconstructed_values, "value")
        return reconstructed_keys, reconstructed_values

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

    def _materialize_state(self) -> Tuple[mx.array, mx.array]:
        key_parts = []
        value_parts = []
        if self._key_chunks:
            key_parts.append(self._decode_all_chunks(self._key_chunks))
            value_parts.append(self._decode_all_chunks(self._value_chunks))
        if self._recent_keys is not None:
            key_parts.append(self._recent_keys)
            value_parts.append(self._recent_values)
        if not key_parts:
            empty = mx.zeros((0,), dtype=mx.float32)
            return empty, empty
        if len(key_parts) == 1:
            return key_parts[0], value_parts[0]
        return mx.concatenate(key_parts, axis=2), mx.concatenate(value_parts, axis=2)

    def _append_recent_tokens(self, keys: mx.array, values: mx.array, *, start: int) -> None:
        if self.use_dense_shadow:
            self._dense_keys[..., start : start + keys.shape[2], :] = keys
            self._dense_values[..., start : start + values.shape[2], :] = values

        if self._recent_keys is None:
            self._recent_start = start
            self._recent_keys = keys
            self._recent_values = values
        else:
            self._recent_keys = mx.concatenate([self._recent_keys, keys], axis=2)
            self._recent_values = mx.concatenate([self._recent_values, values], axis=2)

        flush_limit = self.recent_window_tokens
        if keys.shape[2] == 1:
            flush_limit += self.recent_slack_tokens
        overflow = self._recent_keys.shape[2] - flush_limit
        if overflow <= 0:
            return

        flush_keys = self._recent_keys[..., :overflow, :]
        flush_values = self._recent_values[..., :overflow, :]
        reconstructed_keys, reconstructed_values = self._encode_and_store(flush_keys, flush_values)

        if self.use_dense_shadow:
            flush_end = self._recent_start + overflow
            self._dense_keys[..., self._recent_start:flush_end, :] = reconstructed_keys
            self._dense_values[..., self._recent_start:flush_end, :] = reconstructed_values

        self._recent_keys = self._recent_keys[..., overflow:, :]
        self._recent_values = self._recent_values[..., overflow:, :]
        self._recent_start += overflow

    def update_and_fetch(self, keys, values):
        prev = self.offset
        self.offset += keys.shape[2]
        if self.use_dense_shadow:
            self._ensure_dense_capacity(prev, keys, values)
        if self.recent_window_tokens > 0:
            self._append_recent_tokens(keys, values, start=prev)
        else:
            reconstructed_keys, reconstructed_values = self._encode_and_store(keys, values)
            if self.use_dense_shadow:
                self._dense_keys[..., prev : self.offset, :] = reconstructed_keys
                self._dense_values[..., prev : self.offset, :] = reconstructed_values

        if self.use_dense_shadow:
            return (
                self._dense_keys[..., : self.offset, :],
                self._dense_values[..., : self.offset, :],
            )
        return self._materialize_state()

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
        return self._materialize_state()

    @state.setter
    def state(self, value) -> None:
        keys, values = value
        self._reset_chunks()
        if keys is None or values is None:
            return
        self.update_and_fetch(keys, values)

    @property
    def dense_nbytes(self) -> int:
        if self._dense_keys is None:
            return 0
        return int(self._dense_keys.nbytes + self._dense_values.nbytes)

    @property
    def index_shadow_nbytes(self) -> int:
        return int(sum(chunk.index_shadow_nbytes for chunk in self._key_chunks + self._value_chunks))

    @property
    def nbytes(self):
        chunk_bytes = cache_packed_nbytes(self)
        quantizer_bytes = cache_quantizer_metadata_nbytes(self)
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


class TurboQuantDirectKVCache:
    def __init__(
        self,
        bits: float,
        seed: int = 0,
        norm_dtype: object = mx.float16,
        compute_stats: bool = False,
        block_size: int = 256,
        recent_window_tokens: int = 0,
        recent_slack_tokens: int = 0,
        use_index_shadow: bool = True,
        use_compressed_prefix_cache: bool = True,
        use_rotated_block_cache: bool = True,
        use_metal_scores: bool = True,
        use_metal_values: bool = True,
        metal_max_query_length: int = 1,
        quantizer_pool: TurboQuantQuantizerPool | None = None,
        lean_mode: bool = False,
    ):
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")

        self.turbo_bits = normalize_bit_width(bits)
        self.seed = seed
        self.norm_dtype = norm_dtype
        self.compute_stats = compute_stats
        self.block_size = block_size
        self.recent_window_tokens = recent_window_tokens
        self.recent_slack_tokens = recent_slack_tokens
        if lean_mode:
            use_index_shadow = False
            use_compressed_prefix_cache = False
            use_rotated_block_cache = False
        self.use_index_shadow = use_index_shadow
        self.use_compressed_prefix_cache = use_compressed_prefix_cache
        self.use_rotated_block_cache = use_rotated_block_cache
        self.lean_mode = lean_mode
        uniform_bits = float(self.turbo_bits).is_integer()
        self.use_metal_scores = (
            uniform_bits and use_metal_scores and has_turboquant_score_kernel()
        )
        self.use_metal_values = (
            uniform_bits and use_metal_values and has_turboquant_value_kernel()
        )
        self.use_metal_fused_attention = (
            self.use_metal_scores
            and self.use_metal_values
            and has_turboquant_fused_attention_kernel()
        )
        self.metal_max_query_length = metal_max_query_length
        self._quantizer_pool = quantizer_pool
        self.offset = 0

        self._blocks: List[_JointEncodedBlock] = []
        self._quantizers: Dict[int, object] = {}
        self._cached_compressed_prefix = None
        self._recent_start = 0
        self._recent_keys = None
        self._recent_keys_rotated = None
        self._recent_values = None
        self._recent_values_rotated = None
        self._reconstruction_stats = {
            "key_sq_error_sum": 0.0,
            "value_sq_error_sum": 0.0,
            "key_cosine_sum": 0.0,
            "value_cosine_sum": 0.0,
            "key_vector_count": 0,
            "value_vector_count": 0,
        }

    def _reset(self) -> None:
        self.offset = 0
        self._blocks = []
        self._cached_compressed_prefix = None
        self._recent_start = 0
        self._recent_keys = None
        self._recent_keys_rotated = None
        self._recent_values = None
        self._recent_values_rotated = None

    def _get_quantizer(self, dimension: int):
        if self._quantizer_pool is not None:
            return self._quantizer_pool.get(
                dimension=dimension,
                bits=self.turbo_bits,
                seed=self.seed,
            )

        quantizers = self._quantizers.get(dimension)
        if quantizers is None:
            quantizers = build_mlx_quantizer(
                dimension=dimension,
                bits=self.turbo_bits,
                seed=self.seed + dimension,
            )
            self._quantizers[dimension] = quantizers
        return quantizers

    def _encode_tensor(
        self,
        values: mx.array,
    ) -> Tuple[object | None, mx.array | None, int, mx.array, Tuple[int, ...], object, mx.array | None]:
        original_dtype = values.dtype
        shape = tuple(values.shape)
        dimension = values.shape[-1]
        flattened = mx.reshape(values.astype(mx.float32), (-1, dimension))
        norms = mx.sqrt(mx.maximum(mx.sum(flattened * flattened, axis=1, keepdims=True), 1e-12))
        safe_norms = mx.maximum(norms, 1e-12)
        unit_vectors = flattened / safe_norms

        mlx_quantizer = self._get_quantizer(dimension)
        indices = mlx_quantizer.quantize_indices(unit_vectors)
        index_shadow = None
        if self.use_index_shadow:
            index_shadow = mx.reshape(indices, values.shape)
        reconstructed = None
        if self.compute_stats:
            reconstructed_unit = mlx_quantizer.dequantize_indices(indices)
            reconstructed = mx.reshape(reconstructed_unit * norms, values.shape).astype(original_dtype)

        norms = mx.reshape(norms.astype(self.norm_dtype), values.shape[:-1] + (1,))
        packed = None
        packed_nbytes = mlx_quantizer.codec.storage_nbytes(shape)
        if index_shadow is None:
            packed = mlx_quantizer.codec.pack(mx.reshape(indices, shape))
        return packed, index_shadow, packed_nbytes, norms, shape, original_dtype, reconstructed

    def _decode_tensor(
        self,
        packed_indices,
        norms: mx.array,
        shape: Tuple[int, ...],
        original_dtype: object,
    ) -> mx.array:
        dimension = shape[-1]
        mlx_quantizer = self._get_quantizer(dimension)
        indices = mlx_quantizer.codec.unpack(packed_indices, shape)
        reconstructed_unit = mlx_quantizer.dequantize_indices(mx.reshape(indices, (-1, dimension)))
        reconstructed = mx.reshape(reconstructed_unit, shape) * norms.astype(mx.float32)
        return reconstructed.astype(original_dtype)

    def _materialize_block_key_packed_indices(self, block: _JointEncodedBlock):
        if block.key_packed_indices is None:
            if block.key_indices is None:
                raise ValueError("Encoded key block is missing both packed and unpacked indices.")
            key_quantizer = self._get_quantizer(block.key_shape[-1])
            block.key_packed_indices = key_quantizer.codec.pack(block.key_indices)
        return block.key_packed_indices

    def _materialize_block_value_packed_indices(self, block: _JointEncodedBlock):
        if block.value_packed_indices is None:
            if block.value_indices is None:
                raise ValueError("Encoded value block is missing both packed and unpacked indices.")
            value_quantizer = self._get_quantizer(block.value_shape[-1])
            block.value_packed_indices = value_quantizer.codec.pack(block.value_indices)
        return block.value_packed_indices

    def _compressed_prefix_tensors(
        self,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array] | None:
        if not self.use_compressed_prefix_cache:
            return None
        if not self._blocks:
            return None
        if self._cached_compressed_prefix is not None:
            return self._cached_compressed_prefix
        if any(block.key_indices is None or block.value_indices is None for block in self._blocks):
            return None
        if len(self._blocks) == 1:
            block = self._blocks[0]
            return (
                block.key_indices,
                block.key_norms,
                block.value_indices,
                block.value_norms,
            )
        self._cached_compressed_prefix = (
            mx.concatenate([block.key_indices for block in self._blocks], axis=2),
            mx.concatenate([block.key_norms for block in self._blocks], axis=2),
            mx.concatenate([block.value_indices for block in self._blocks], axis=2),
            mx.concatenate([block.value_norms for block in self._blocks], axis=2),
        )
        return self._cached_compressed_prefix

    def _append_stats(self, original: mx.array, reconstructed: mx.array | None, kind: str) -> None:
        if not self.compute_stats or reconstructed is None:
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

    def _append_compressed_tokens(
        self,
        keys: mx.array,
        values: mx.array,
        *,
        start: int,
    ) -> None:
        num_tokens = keys.shape[2]
        for block_offset in range(0, num_tokens, self.block_size):
            block_end = min(block_offset + self.block_size, num_tokens)
            key_block = keys[..., block_offset:block_end, :]
            value_block = values[..., block_offset:block_end, :]

            (
                key_packed,
                key_indices,
                key_packed_nbytes,
                key_norms,
                key_shape,
                key_dtype,
                key_reconstructed,
            ) = self._encode_tensor(key_block)
            (
                value_packed,
                value_indices,
                value_packed_nbytes,
                value_norms,
                value_shape,
                value_dtype,
                value_reconstructed,
            ) = self._encode_tensor(value_block)

            self._blocks.append(
                _JointEncodedBlock(
                    key_packed_indices=key_packed,
                    key_indices=key_indices,
                    key_packed_nbytes=key_packed_nbytes,
                    key_norms=key_norms,
                    key_shape=key_shape,
                    key_dtype=key_dtype,
                    value_packed_indices=value_packed,
                    value_indices=value_indices,
                    value_packed_nbytes=value_packed_nbytes,
                    value_norms=value_norms,
                    value_shape=value_shape,
                    value_dtype=value_dtype,
                    start=start + block_offset,
                )
            )
            self._cached_compressed_prefix = None

            self._append_stats(key_block, key_reconstructed, "key")
            self._append_stats(value_block, value_reconstructed, "value")

    def _append_recent_tokens(self, keys: mx.array, values: mx.array, *, start: int) -> None:
        key_dim = keys.shape[-1]
        value_dim = values.shape[-1]
        key_quantizer = self._get_quantizer(key_dim)
        value_quantizer = self._get_quantizer(value_dim)
        rotated_keys = key_quantizer.rotate(keys.astype(mx.float32))
        rotated_values = value_quantizer.rotate(values.astype(mx.float32))

        if self._recent_keys is None:
            self._recent_start = start
            self._recent_keys = keys
            self._recent_keys_rotated = rotated_keys
            self._recent_values = values
            self._recent_values_rotated = rotated_values
        else:
            self._recent_keys = mx.concatenate([self._recent_keys, keys], axis=2)
            self._recent_keys_rotated = mx.concatenate(
                [self._recent_keys_rotated, rotated_keys], axis=2
            )
            self._recent_values = mx.concatenate([self._recent_values, values], axis=2)
            self._recent_values_rotated = mx.concatenate(
                [self._recent_values_rotated, rotated_values], axis=2
            )

        flush_limit = self.recent_window_tokens
        if keys.shape[2] == 1:
            flush_limit += self.recent_slack_tokens
        overflow = self._recent_keys.shape[2] - flush_limit
        if overflow <= 0:
            return

        flush_keys = self._recent_keys[..., :overflow, :]
        flush_values = self._recent_values[..., :overflow, :]
        self._append_compressed_tokens(flush_keys, flush_values, start=self._recent_start)
        self._recent_keys = self._recent_keys[..., overflow:, :]
        self._recent_keys_rotated = self._recent_keys_rotated[..., overflow:, :]
        self._recent_values = self._recent_values[..., overflow:, :]
        self._recent_values_rotated = self._recent_values_rotated[..., overflow:, :]
        self._recent_start += overflow

    def append(self, keys: mx.array, values: mx.array) -> int:
        if keys.shape[2] != values.shape[2]:
            raise ValueError("keys and values must have the same sequence length.")

        query_start = self.offset
        if self.recent_window_tokens > 0:
            self._append_recent_tokens(keys, values, start=query_start)
        else:
            self._append_compressed_tokens(keys, values, start=query_start)

        self.offset += keys.shape[2]
        return query_start

    def _rotated_keys_for_block(self, block: _JointEncodedBlock, key_quantizer) -> mx.array:
        if self.use_rotated_block_cache and block.key_rotated_cache is not None:
            return block.key_rotated_cache
        key_indices = block.key_indices
        if key_indices is None:
            key_indices = key_quantizer.codec.unpack(
                self._materialize_block_key_packed_indices(block),
                block.key_shape,
            )
        rotated_keys = key_quantizer.lookup_centroids(key_indices)
        rotated_keys = rotated_keys * block.key_norms.astype(mx.float32)
        if self.use_rotated_block_cache:
            block.key_rotated_cache = rotated_keys
        return rotated_keys

    def _rotated_values_for_block(
        self, block: _JointEncodedBlock, value_quantizer
    ) -> mx.array:
        if self.use_rotated_block_cache and block.value_rotated_cache is not None:
            return block.value_rotated_cache
        value_indices = block.value_indices
        if value_indices is None:
            value_indices = value_quantizer.codec.unpack(
                self._materialize_block_value_packed_indices(block),
                block.value_shape,
            )
        rotated_values = value_quantizer.lookup_centroids(value_indices)
        rotated_values = rotated_values * block.value_norms.astype(mx.float32)
        if self.use_rotated_block_cache:
            block.value_rotated_cache = rotated_values
        return rotated_values

    def _key_indices_for_block(self, block: _JointEncodedBlock, key_quantizer) -> mx.array:
        if block.key_indices is not None:
            return block.key_indices
        return key_quantizer.codec.unpack(
            self._materialize_block_key_packed_indices(block),
            block.key_shape,
        )

    def _value_indices_for_block(
        self, block: _JointEncodedBlock, value_quantizer
    ) -> mx.array:
        if block.value_indices is not None:
            return block.value_indices
        return value_quantizer.codec.unpack(
            self._materialize_block_value_packed_indices(block),
            block.value_shape,
        )

    def _apply_range_mask(
        self,
        scores: mx.array,
        mask,
        *,
        query_start: int,
        query_length: int,
        key_start: int,
        key_length: int,
    ) -> mx.array:
        if mask is None:
            return scores

        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unsupported mask specifier: {mask}")
            q_positions = mx.arange(query_start, query_start + query_length, dtype=mx.int32)
            k_positions = mx.arange(key_start, key_start + key_length, dtype=mx.int32)
            block_mask = q_positions[:, None] >= k_positions[None, :]
            block_mask = mx.reshape(block_mask, (1, 1, 1, query_length, key_length))
            return mx.where(
                block_mask,
                scores,
                mx.full(scores.shape, mx.finfo(scores.dtype).min, dtype=scores.dtype),
            )

        block_mask = mask[..., key_start : key_start + key_length]
        if block_mask.ndim == 2:
            block_mask = mx.reshape(block_mask, (1, 1, 1, query_length, key_length))
        elif block_mask.ndim == 4:
            block_mask = mx.expand_dims(block_mask, axis=2)
        while block_mask.ndim < scores.ndim:
            block_mask = mx.expand_dims(block_mask, axis=0)

        if block_mask.dtype == mx.bool_:
            return mx.where(
                block_mask,
                scores,
                mx.full(scores.shape, mx.finfo(scores.dtype).min, dtype=scores.dtype),
            )
        return scores + block_mask.astype(scores.dtype)

    def direct_attention(
        self,
        queries: mx.array,
        *,
        scale: float,
        mask=None,
        query_start: int,
    ) -> mx.array:
        if not self._blocks and self._recent_keys is None:
            raise ValueError("Cannot attend with an empty cache.")

        if self._blocks:
            key_dim = self._blocks[0].key_shape[-1]
            value_dim = self._blocks[0].value_shape[-1]
            n_kv_heads = self._blocks[0].key_shape[1]
        else:
            key_dim = self._recent_keys.shape[-1]
            value_dim = self._recent_values.shape[-1]
            n_kv_heads = self._recent_keys.shape[1]

        key_quantizer = self._get_quantizer(key_dim)
        value_quantizer = self._get_quantizer(value_dim)

        B, n_heads, query_length, _ = queries.shape
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        n_repeats = n_heads // n_kv_heads

        rotated_queries = key_quantizer.rotate(queries.astype(mx.float32)) * scale
        rotated_queries = mx.reshape(
            rotated_queries, (B, n_kv_heads, n_repeats, query_length, key_dim)
        )

        running_max = None
        running_sum = None
        running_output = None

        compressed_prefix = None
        if (
            self.use_metal_fused_attention
            and query_length == 1
            and (mask is None or mask == "causal")
        ):
            compressed_prefix = self._compressed_prefix_tensors()

        if compressed_prefix is not None:
            (
                compressed_key_indices,
                compressed_key_norms,
                compressed_value_indices,
                compressed_value_norms,
            ) = compressed_prefix
            running_output, running_max, running_sum = turboquant_fused_block_attention(
                mx.reshape(rotated_queries, (B, n_kv_heads, n_repeats, key_dim)),
                compressed_key_indices,
                compressed_key_norms,
                key_quantizer.centroids,
                None,
                None,
                compressed_value_indices,
                compressed_value_norms,
                value_quantizer.centroids,
            )
        else:
            fused_query_positions = None
            if mask == "causal":
                fused_query_positions = mx.broadcast_to(
                    mx.reshape(
                        mx.arange(query_start, query_start + query_length, dtype=mx.int32),
                        (1, 1, 1, query_length),
                    ),
                    (B, n_kv_heads, n_repeats, query_length),
                )
            for block in self._blocks:
                key_indices = None
                value_indices = None
                if self.use_metal_scores or self.use_metal_fused_attention:
                    key_indices = self._key_indices_for_block(block, key_quantizer)
                if self.use_metal_values or self.use_metal_fused_attention:
                    value_indices = self._value_indices_for_block(block, value_quantizer)
                block_supports_fused = (
                    self.use_metal_fused_attention
                    and key_indices is not None
                    and value_indices is not None
                    and query_length <= self.metal_max_query_length
                    and (mask is None or mask == "causal")
                )
                if block_supports_fused:
                    key_positions = None
                    causal = False
                    if mask == "causal":
                        key_positions = mx.arange(
                            block.start,
                            block.start + block.length,
                            dtype=mx.int32,
                        )
                        causal = True
                    block_output, block_max, block_sum = turboquant_fused_block_attention(
                        rotated_queries,
                        key_indices,
                        block.key_norms,
                        key_quantizer.centroids,
                        fused_query_positions,
                        key_positions,
                        value_indices,
                        block.value_norms,
                        value_quantizer.centroids,
                        causal=causal,
                    )
                else:
                    if self.use_metal_scores and query_length <= self.metal_max_query_length:
                        scores = turboquant_score_block(
                            rotated_queries,
                            key_indices,
                            block.key_norms,
                            key_quantizer.centroids,
                        )
                    else:
                        rotated_keys = self._rotated_keys_for_block(block, key_quantizer)
                        scores = mx.matmul(
                            rotated_queries,
                            mx.expand_dims(rotated_keys.transpose(0, 1, 3, 2), axis=2),
                        )
                    scores = self._apply_range_mask(
                        scores,
                        mask,
                        query_start=query_start,
                        query_length=query_length,
                        key_start=block.start,
                        key_length=block.length,
                    )

                    block_max = mx.max(scores, axis=-1, keepdims=True)
                    weights = mx.exp(scores - block_max)
                    block_sum = mx.sum(weights, axis=-1, keepdims=True)

                    if self.use_metal_values and query_length <= self.metal_max_query_length:
                        block_output = turboquant_value_block(
                            weights,
                            value_indices,
                            block.value_norms,
                            value_quantizer.centroids,
                        )
                    else:
                        rotated_values = self._rotated_values_for_block(block, value_quantizer)
                        block_output = mx.matmul(
                            weights,
                            mx.expand_dims(rotated_values, axis=2),
                        )

                if running_max is None:
                    running_max = block_max
                    running_sum = block_sum
                    running_output = block_output
                else:
                    new_max = mx.maximum(running_max, block_max)
                    existing_scale = mx.exp(running_max - new_max)
                    block_scale = mx.exp(block_max - new_max)
                    running_sum = running_sum * existing_scale + block_sum * block_scale
                    running_output = (
                        running_output * existing_scale + block_output * block_scale
                    )
                    running_max = new_max

        if self._recent_keys is not None and self._recent_keys.shape[2] > 0:
            recent_scores = mx.matmul(
                rotated_queries,
                mx.expand_dims(
                    self._recent_keys_rotated.transpose(0, 1, 3, 2),
                    axis=2,
                ),
            )
            recent_scores = self._apply_range_mask(
                recent_scores,
                mask,
                query_start=query_start,
                query_length=query_length,
                key_start=self._recent_start,
                key_length=self._recent_keys.shape[2],
            )

            recent_max = mx.max(recent_scores, axis=-1, keepdims=True)
            recent_weights = mx.exp(recent_scores - recent_max)
            recent_sum = mx.sum(recent_weights, axis=-1, keepdims=True)
            recent_output = mx.matmul(
                recent_weights,
                mx.expand_dims(self._recent_values_rotated, axis=2),
            )

            if running_max is None:
                running_max = recent_max
                running_sum = recent_sum
                running_output = recent_output
            else:
                new_max = mx.maximum(running_max, recent_max)
                existing_scale = mx.exp(running_max - new_max)
                recent_scale = mx.exp(recent_max - new_max)
                running_sum = running_sum * existing_scale + recent_sum * recent_scale
                running_output = (
                    running_output * existing_scale + recent_output * recent_scale
                )
                running_max = new_max

        output_rotated = running_output / mx.maximum(running_sum, 1e-12)
        output_rotated = mx.reshape(output_rotated, (B, n_heads, query_length, value_dim))
        return value_quantizer.unrotate(output_rotated).astype(queries.dtype)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self.append(keys, values)
        return self.decoded_state

    @property
    def decoded_state(self) -> Tuple[mx.array, mx.array]:
        if not self._blocks and self._recent_keys is None:
            return mx.zeros((0,), dtype=mx.float32), mx.zeros((0,), dtype=mx.float32)
        keys = [
            self._decode_tensor(
                self._materialize_block_key_packed_indices(block),
                block.key_norms,
                block.key_shape,
                block.key_dtype,
            )
            for block in self._blocks
        ]
        values = [
            self._decode_tensor(
                self._materialize_block_value_packed_indices(block),
                block.value_norms,
                block.value_shape,
                block.value_dtype,
            )
            for block in self._blocks
        ]
        if self._recent_keys is not None:
            keys.append(self._recent_keys)
            values.append(self._recent_values)
        return mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.offset == 0

    @property
    def state(self):
        if not self._blocks and self._recent_keys is None:
            return (mx.zeros((0,), dtype=mx.uint32),)
        items = []
        for block in self._blocks:
            items.extend(
                [
                    self._materialize_block_key_packed_indices(block),
                    block.key_norms,
                    self._materialize_block_value_packed_indices(block),
                    block.value_norms,
                ]
            )
        if self._recent_keys is not None:
            items.extend(
                [
                    self._recent_keys,
                    self._recent_keys_rotated,
                    self._recent_values,
                    self._recent_values_rotated,
                ]
            )
        return tuple(items)

    @state.setter
    def state(self, value) -> None:
        self._reset()
        if value is None:
            return
        if isinstance(value, (tuple, list)) and len(value) == 0:
            return
        raise ValueError("Setting state is not supported for TurboQuantDirectKVCache.")

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self.turbo_bits,
                    self.block_size,
                    self.recent_window_tokens,
                    self.recent_slack_tokens,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value) -> None:
        items = list(value)
        self.offset = int(items[0])
        self.turbo_bits = normalize_bit_width(float(items[1]))
        self.block_size = int(items[2])
        self.recent_window_tokens = int(items[3]) if len(items) > 3 else 0
        self.recent_slack_tokens = int(items[4]) if len(items) > 4 else 0
        uniform_bits = float(self.turbo_bits).is_integer()
        self.use_metal_scores = uniform_bits and has_turboquant_score_kernel()
        self.use_metal_values = uniform_bits and has_turboquant_value_kernel()
        self.use_metal_fused_attention = (
            self.use_metal_scores
            and self.use_metal_values
            and has_turboquant_fused_attention_kernel()
        )

    @property
    def dense_nbytes(self) -> int:
        if self._recent_keys_rotated is None or self._recent_values_rotated is None:
            return 0
        return int(self._recent_keys_rotated.nbytes + self._recent_values_rotated.nbytes)

    @property
    def index_shadow_nbytes(self) -> int:
        total = 0
        for block in self._blocks:
            if block.key_indices is not None:
                total += int(block.key_indices.nbytes)
            if block.value_indices is not None:
                total += int(block.value_indices.nbytes)
        return total

    @property
    def prefix_shadow_nbytes(self) -> int:
        if self._cached_compressed_prefix is None:
            return 0
        return int(sum(item.nbytes for item in self._cached_compressed_prefix))

    @property
    def rotated_block_cache_nbytes(self) -> int:
        return int(sum(block.rotated_cache_nbytes for block in self._blocks))

    @property
    def nbytes(self):
        block_bytes = sum(block.storage_nbytes for block in self._blocks)
        recent_bytes = 0
        if self._recent_keys is not None:
            recent_bytes += int(self._recent_keys.nbytes + self._recent_values.nbytes)
        quantizer_bytes = cache_quantizer_metadata_nbytes(self)
        return int(block_bytes + recent_bytes + quantizer_bytes)

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
