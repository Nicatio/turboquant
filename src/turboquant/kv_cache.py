from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from turboquant.metal_kernels import has_turboquant_score_kernel, turboquant_score_block
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


@dataclass
class _JointEncodedBlock:
    key_packed_indices: mx.array
    key_indices: mx.array | None
    key_norms: mx.array
    key_shape: Tuple[int, ...]
    key_dtype: object
    value_packed_indices: mx.array
    value_indices: mx.array | None
    value_norms: mx.array
    value_shape: Tuple[int, ...]
    value_dtype: object
    start: int

    @property
    def length(self) -> int:
        return self.key_shape[2]

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.key_packed_indices.nbytes
            + self.key_norms.nbytes
            + self.value_packed_indices.nbytes
            + self.value_norms.nbytes
        )


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


class TurboQuantDirectKVCache:
    def __init__(
        self,
        bits: int,
        seed: int = 0,
        norm_dtype: object = mx.float16,
        compute_stats: bool = False,
        block_size: int = 256,
        use_index_shadow: bool = True,
        use_metal_scores: bool = True,
        metal_max_query_length: int = 1,
    ):
        if bits < 1:
            raise ValueError("bits must be at least 1.")
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")

        self.turbo_bits = bits
        self.seed = seed
        self.norm_dtype = norm_dtype
        self.compute_stats = compute_stats
        self.block_size = block_size
        self.use_index_shadow = use_index_shadow
        self.use_metal_scores = use_metal_scores and has_turboquant_score_kernel()
        self.metal_max_query_length = metal_max_query_length
        self.offset = 0

        self._blocks: List[_JointEncodedBlock] = []
        self._quantizers: Dict[int, Tuple[TurboQuantMSE, MlxTurboQuantMSE]] = {}
        self._codec = PackedIndexCodec(bits=bits)
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

    def _encode_tensor(
        self,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array | None, mx.array, Tuple[int, ...], object, mx.array | None]:
        original_dtype = values.dtype
        dimension = values.shape[-1]
        flattened = mx.reshape(values.astype(mx.float32), (-1, dimension))
        norms = mx.sqrt(mx.maximum(mx.sum(flattened * flattened, axis=1, keepdims=True), 1e-12))
        safe_norms = mx.maximum(norms, 1e-12)
        unit_vectors = flattened / safe_norms

        _, mlx_quantizer = self._get_quantizer(dimension)
        indices = mlx_quantizer.quantize_indices(unit_vectors)
        index_shadow = None
        if self.use_index_shadow:
            index_shadow = mx.reshape(indices, values.shape)
        reconstructed = None
        if self.compute_stats:
            reconstructed_unit = mlx_quantizer.dequantize_indices(indices)
            reconstructed = mx.reshape(reconstructed_unit * norms, values.shape).astype(original_dtype)

        norms = mx.reshape(norms.astype(self.norm_dtype), values.shape[:-1] + (1,))
        packed = self._codec.pack(mx.reshape(indices, values.shape))
        return packed, index_shadow, norms, tuple(values.shape), original_dtype, reconstructed

    def _decode_tensor(
        self,
        packed_indices: mx.array,
        norms: mx.array,
        shape: Tuple[int, ...],
        original_dtype: object,
    ) -> mx.array:
        dimension = shape[-1]
        _, mlx_quantizer = self._get_quantizer(dimension)
        indices = self._codec.unpack(packed_indices, shape)
        rotated = mlx_quantizer.lookup_centroids(indices)
        reconstructed = mlx_quantizer.unrotate(rotated * norms.astype(mx.float32))
        return reconstructed.astype(original_dtype)

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

    def append(self, keys: mx.array, values: mx.array) -> int:
        if keys.shape[2] != values.shape[2]:
            raise ValueError("keys and values must have the same sequence length.")

        query_start = self.offset
        num_tokens = keys.shape[2]

        for block_offset in range(0, num_tokens, self.block_size):
            block_end = min(block_offset + self.block_size, num_tokens)
            key_block = keys[..., block_offset:block_end, :]
            value_block = values[..., block_offset:block_end, :]

            (
                key_packed,
                key_indices,
                key_norms,
                key_shape,
                key_dtype,
                key_reconstructed,
            ) = self._encode_tensor(key_block)
            (
                value_packed,
                value_indices,
                value_norms,
                value_shape,
                value_dtype,
                value_reconstructed,
            ) = self._encode_tensor(value_block)

            self._blocks.append(
                _JointEncodedBlock(
                    key_packed_indices=key_packed,
                    key_indices=key_indices,
                    key_norms=key_norms,
                    key_shape=key_shape,
                    key_dtype=key_dtype,
                    value_packed_indices=value_packed,
                    value_indices=value_indices,
                    value_norms=value_norms,
                    value_shape=value_shape,
                    value_dtype=value_dtype,
                    start=query_start + block_offset,
                )
            )

            self._append_stats(key_block, key_reconstructed, "key")
            self._append_stats(value_block, value_reconstructed, "value")

        self.offset += num_tokens
        return query_start

    def _apply_block_mask(
        self,
        scores: mx.array,
        mask,
        *,
        query_start: int,
        query_length: int,
        block: _JointEncodedBlock,
    ) -> mx.array:
        if mask is None:
            return scores

        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unsupported mask specifier: {mask}")
            q_positions = mx.arange(query_start, query_start + query_length, dtype=mx.int32)
            k_positions = mx.arange(block.start, block.start + block.length, dtype=mx.int32)
            block_mask = q_positions[:, None] >= k_positions[None, :]
            block_mask = mx.reshape(block_mask, (1, 1, 1, query_length, block.length))
            return mx.where(
                block_mask,
                scores,
                mx.full(scores.shape, mx.finfo(scores.dtype).min, dtype=scores.dtype),
            )

        block_mask = mask[..., block.start : block.start + block.length]
        if block_mask.ndim == 2:
            block_mask = mx.reshape(block_mask, (1, 1, 1, query_length, block.length))
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
        if not self._blocks:
            raise ValueError("Cannot attend with an empty cache.")

        key_dim = self._blocks[0].key_shape[-1]
        value_dim = self._blocks[0].value_shape[-1]
        _, key_quantizer = self._get_quantizer(key_dim)
        _, value_quantizer = self._get_quantizer(value_dim)

        B, n_heads, query_length, _ = queries.shape
        n_kv_heads = self._blocks[0].key_shape[1]
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

        for block in self._blocks:
            key_indices = block.key_indices
            if (
                self.use_metal_scores
                and key_indices is not None
                and query_length <= self.metal_max_query_length
            ):
                scores = turboquant_score_block(
                    rotated_queries,
                    key_indices,
                    block.key_norms,
                    key_quantizer.centroids,
                )
            else:
                if key_indices is None:
                    key_indices = self._codec.unpack(block.key_packed_indices, block.key_shape)
                rotated_keys = key_quantizer.lookup_centroids(key_indices)
                rotated_keys = rotated_keys * block.key_norms.astype(mx.float32)
                scores = mx.matmul(
                    rotated_queries,
                    mx.expand_dims(rotated_keys.transpose(0, 1, 3, 2), axis=2),
                )
            scores = self._apply_block_mask(
                scores,
                mask,
                query_start=query_start,
                query_length=query_length,
                block=block,
            )

            block_max = mx.max(scores, axis=-1, keepdims=True)
            weights = mx.exp(scores - block_max)
            block_sum = mx.sum(weights, axis=-1, keepdims=True)

            value_indices = block.value_indices
            if value_indices is None:
                value_indices = self._codec.unpack(block.value_packed_indices, block.value_shape)
            rotated_values = value_quantizer.lookup_centroids(value_indices)
            rotated_values = rotated_values * block.value_norms.astype(mx.float32)
            block_output = mx.matmul(weights, mx.expand_dims(rotated_values, axis=2))

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

        output_rotated = running_output / mx.maximum(running_sum, 1e-12)
        output_rotated = mx.reshape(output_rotated, (B, n_heads, query_length, value_dim))
        return value_quantizer.unrotate(output_rotated).astype(queries.dtype)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self.append(keys, values)
        return self.decoded_state

    @property
    def decoded_state(self) -> Tuple[mx.array, mx.array]:
        if not self._blocks:
            return mx.zeros((0,), dtype=mx.float32), mx.zeros((0,), dtype=mx.float32)
        keys = [
            self._decode_tensor(
                block.key_packed_indices,
                block.key_norms,
                block.key_shape,
                block.key_dtype,
            )
            for block in self._blocks
        ]
        values = [
            self._decode_tensor(
                block.value_packed_indices,
                block.value_norms,
                block.value_shape,
                block.value_dtype,
            )
            for block in self._blocks
        ]
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
        if not self._blocks:
            return (mx.zeros((0,), dtype=mx.uint32),)
        items = []
        for block in self._blocks:
            items.extend(
                [
                    block.key_packed_indices,
                    block.key_norms,
                    block.value_packed_indices,
                    block.value_norms,
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
        return tuple(map(str, (self.offset, self.turbo_bits, self.block_size)))

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.offset, self.turbo_bits, self.block_size = map(int, value)

    @property
    def dense_nbytes(self) -> int:
        return 0

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
    def nbytes(self):
        block_bytes = sum(block.storage_nbytes for block in self._blocks)
        quantizer_bytes = 0
        for numpy_quantizer, _ in self._quantizers.values():
            quantizer_bytes += numpy_quantizer.rotation.matrix.nbytes
            quantizer_bytes += numpy_quantizer.codebook.centroids.nbytes
            quantizer_bytes += numpy_quantizer.codebook.thresholds.nbytes
        return int(block_bytes + quantizer_bytes)

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
