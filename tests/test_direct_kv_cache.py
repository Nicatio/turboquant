from __future__ import annotations

import pathlib
import sys
import unittest

import mlx.core as mx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.kv_cache import (
    TurboQuantDirectKVCache,
    TurboQuantQuantizerPool,
    cache_list_nbytes,
)
from turboquant.metal_kernels import (
    has_turboquant_fused_attention_kernel,
    has_turboquant_score_kernel,
    has_turboquant_value_kernel,
    turboquant_fused_block_attention,
    turboquant_score_block,
    turboquant_value_block,
)


def dense_reference_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    scale: float,
    *,
    query_start: int,
    causal: bool,
) -> np.ndarray:
    batch, n_heads, query_length, head_dim = queries.shape
    _, n_kv_heads, total_tokens, _ = keys.shape
    n_repeats = n_heads // n_kv_heads

    output = np.zeros((batch, n_heads, query_length, head_dim), dtype=np.float32)
    for b in range(batch):
        for head in range(n_heads):
            kv_head = head // n_repeats
            q = queries[b, head]
            k = keys[b, kv_head]
            v = values[b, kv_head]
            scores = (q @ k.T) * scale
            if causal:
                q_positions = np.arange(query_start, query_start + query_length)[:, None]
                k_positions = np.arange(total_tokens)[None, :]
                scores = np.where(q_positions >= k_positions, scores, -1e30)
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= np.sum(weights, axis=-1, keepdims=True)
            output[b, head] = weights @ v
    return output


class TurboQuantDirectKVCacheTests(unittest.TestCase):
    def test_metal_score_kernel_matches_reference(self) -> None:
        if not has_turboquant_score_kernel():
            self.skipTest("Metal score kernel is unavailable.")

        rng = np.random.default_rng(4)
        queries = rng.standard_normal((1, 2, 3, 2, 8)).astype(np.float32)
        key_indices = rng.integers(0, 8, size=(1, 2, 5, 8), dtype=np.uint8)
        key_norms = rng.random((1, 2, 5, 1), dtype=np.float32)
        centroids = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)

        actual = np.asarray(
            turboquant_score_block(
                mx.array(queries),
                mx.array(key_indices),
                mx.array(key_norms),
                mx.array(centroids),
            )
        )

        rotated_keys = centroids[key_indices.astype(np.int32)] * key_norms
        expected = np.einsum("bhrld,bhtd->bhrlt", queries, rotated_keys, optimize=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_direct_attention_matches_dense_reconstruction(self) -> None:
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        queries = rng.standard_normal((1, 4, 5, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3, seed=0, block_size=2)
        query_start = cache.append(mx.array(keys), mx.array(values))
        actual = np.asarray(
            cache.direct_attention(
                mx.array(queries),
                scale=8**-0.5,
                mask="causal",
                query_start=query_start,
            )
        )

        decoded_keys, decoded_values = cache.decoded_state
        expected = dense_reference_attention(
            queries,
            np.asarray(decoded_keys),
            np.asarray(decoded_values),
            8**-0.5,
            query_start=query_start,
            causal=True,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_metal_value_kernel_matches_reference(self) -> None:
        if not has_turboquant_value_kernel():
            self.skipTest("Metal value kernel is unavailable.")

        rng = np.random.default_rng(5)
        weights = rng.random((1, 2, 3, 2, 5), dtype=np.float32)
        value_indices = rng.integers(0, 8, size=(1, 2, 5, 8), dtype=np.uint8)
        value_norms = rng.random((1, 2, 5, 1), dtype=np.float32)
        centroids = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)

        actual = np.asarray(
            turboquant_value_block(
                mx.array(weights),
                mx.array(value_indices),
                mx.array(value_norms),
                mx.array(centroids),
            )
        )

        rotated_values = centroids[value_indices.astype(np.int32)] * value_norms
        expected = np.einsum("bhrlt,bhtd->bhrld", weights, rotated_values, optimize=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_metal_fused_attention_kernel_matches_reference(self) -> None:
        if not has_turboquant_fused_attention_kernel():
            self.skipTest("Metal fused attention kernel is unavailable.")

        rng = np.random.default_rng(12)
        rotated_queries = rng.standard_normal((1, 2, 3, 8)).astype(np.float32)
        key_indices = rng.integers(0, 8, size=(1, 2, 5, 8), dtype=np.uint8)
        key_norms = rng.random((1, 2, 5, 1), dtype=np.float32)
        value_indices = rng.integers(0, 8, size=(1, 2, 5, 8), dtype=np.uint8)
        value_norms = rng.random((1, 2, 5, 1), dtype=np.float32)
        key_centroids = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
        value_centroids = np.linspace(-0.5, 0.5, num=8, dtype=np.float32)

        actual_output, actual_max, actual_sum = turboquant_fused_block_attention(
            mx.array(rotated_queries),
            mx.array(key_indices),
            mx.array(key_norms),
            mx.array(key_centroids),
            None,
            None,
            mx.array(value_indices),
            mx.array(value_norms),
            mx.array(value_centroids),
        )
        actual_output = np.asarray(actual_output)
        actual_max = np.asarray(actual_max)
        actual_sum = np.asarray(actual_sum)

        rotated_keys = key_centroids[key_indices.astype(np.int32)] * key_norms
        scores = np.einsum("bhrd,bhtd->bhrt", rotated_queries, rotated_keys, optimize=True)[
            ..., None, :
        ]
        expected_max = np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores - expected_max)
        expected_sum = np.sum(weights, axis=-1, keepdims=True)
        rotated_values = value_centroids[value_indices.astype(np.int32)] * value_norms
        expected_output = np.einsum("bhrlt,bhtd->bhrld", weights, rotated_values, optimize=True)

        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_max, expected_max, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-5, atol=1e-5)

    def test_metal_fused_attention_kernel_matches_causal_multiquery_reference(self) -> None:
        if not has_turboquant_fused_attention_kernel():
            self.skipTest("Metal fused attention kernel is unavailable.")

        rng = np.random.default_rng(16)
        rotated_queries = rng.standard_normal((1, 2, 3, 4, 8)).astype(np.float32)
        key_indices = rng.integers(0, 8, size=(1, 2, 5, 8), dtype=np.uint8)
        key_norms = rng.random((1, 2, 5, 1), dtype=np.float32)
        value_indices = rng.integers(0, 8, size=(1, 2, 5, 8), dtype=np.uint8)
        value_norms = rng.random((1, 2, 5, 1), dtype=np.float32)
        key_centroids = np.linspace(-1.0, 1.0, num=8, dtype=np.float32)
        value_centroids = np.linspace(-0.5, 0.5, num=8, dtype=np.float32)
        query_positions = np.broadcast_to(
            np.arange(4, dtype=np.int32).reshape(1, 1, 1, 4),
            (1, 2, 3, 4),
        )
        key_positions = np.arange(5, dtype=np.int32)

        actual_output, actual_max, actual_sum = turboquant_fused_block_attention(
            mx.array(rotated_queries),
            mx.array(key_indices),
            mx.array(key_norms),
            mx.array(key_centroids),
            mx.array(query_positions),
            mx.array(key_positions),
            mx.array(value_indices),
            mx.array(value_norms),
            mx.array(value_centroids),
            causal=True,
        )
        actual_output = np.asarray(actual_output)
        actual_max = np.asarray(actual_max)
        actual_sum = np.asarray(actual_sum)

        rotated_keys = key_centroids[key_indices.astype(np.int32)] * key_norms
        scores = np.einsum(
            "bhrld,bhtd->bhrlt",
            rotated_queries,
            rotated_keys,
            optimize=True,
        )
        mask = query_positions[..., None] >= key_positions.reshape(1, 1, 1, 1, 5)
        scores = np.where(mask, scores, -np.inf)
        expected_max = np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores - expected_max)
        expected_sum = np.sum(weights, axis=-1, keepdims=True)
        rotated_values = value_centroids[value_indices.astype(np.int32)] * value_norms
        expected_output = np.einsum("bhrlt,bhtd->bhrld", weights, rotated_values, optimize=True)

        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_max, expected_max, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-5, atol=1e-5)

    def test_incremental_direct_attention_matches_dense_reconstruction(self) -> None:
        rng = np.random.default_rng(1)
        prefix_keys = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
        prefix_values = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
        next_keys = rng.standard_normal((1, 2, 1, 8)).astype(np.float32)
        next_values = rng.standard_normal((1, 2, 1, 8)).astype(np.float32)
        next_queries = rng.standard_normal((1, 4, 1, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3, seed=1, block_size=2)
        cache.append(mx.array(prefix_keys), mx.array(prefix_values))
        query_start = cache.append(mx.array(next_keys), mx.array(next_values))
        actual = np.asarray(
            cache.direct_attention(
                mx.array(next_queries),
                scale=8**-0.5,
                mask=None,
                query_start=query_start,
            )
        )

        decoded_keys, decoded_values = cache.decoded_state
        expected = dense_reference_attention(
            next_queries,
            np.asarray(decoded_keys),
            np.asarray(decoded_values),
            8**-0.5,
            query_start=query_start,
            causal=False,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_recent_window_hybrid_matches_dense_reconstruction(self) -> None:
        rng = np.random.default_rng(6)
        keys = rng.standard_normal((1, 2, 7, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 7, 8)).astype(np.float32)
        queries = rng.standard_normal((1, 4, 2, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(
            bits=3,
            seed=6,
            block_size=2,
            recent_window_tokens=3,
        )
        cache.append(mx.array(keys), mx.array(values))
        self.assertIsNotNone(cache._recent_keys_rotated)
        self.assertEqual(cache._recent_keys_rotated.shape, cache._recent_keys.shape)
        actual = np.asarray(
            cache.direct_attention(
                mx.array(queries),
                scale=8**-0.5,
                mask=None,
                query_start=5,
            )
        )

        decoded_keys, decoded_values = cache.decoded_state
        expected = dense_reference_attention(
            queries,
            np.asarray(decoded_keys),
            np.asarray(decoded_values),
            8**-0.5,
            query_start=5,
            causal=False,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_recent_slack_defers_single_token_flushes(self) -> None:
        rng = np.random.default_rng(7)
        prefix_keys = mx.array(rng.standard_normal((1, 2, 4, 8)).astype(np.float32))
        prefix_values = mx.array(rng.standard_normal((1, 2, 4, 8)).astype(np.float32))
        token_keys = mx.array(rng.standard_normal((1, 2, 1, 8)).astype(np.float32))
        token_values = mx.array(rng.standard_normal((1, 2, 1, 8)).astype(np.float32))

        cache = TurboQuantDirectKVCache(
            bits=3,
            seed=7,
            block_size=2,
            recent_window_tokens=2,
            recent_slack_tokens=2,
        )
        cache.append(prefix_keys, prefix_values)
        self.assertEqual(len(cache._blocks), 1)
        self.assertEqual(cache._recent_keys.shape[2], 2)

        cache.append(token_keys, token_values)
        self.assertEqual(len(cache._blocks), 1)
        self.assertEqual(cache._recent_keys.shape[2], 3)

        cache.append(token_keys, token_values)
        self.assertEqual(len(cache._blocks), 1)
        self.assertEqual(cache._recent_keys.shape[2], 4)

        cache.append(token_keys, token_values)
        self.assertEqual(len(cache._blocks), 2)
        self.assertEqual(cache._recent_keys.shape[2], 4)

    def test_direct_cache_storage_is_smaller_than_float32_cache(self) -> None:
        rng = np.random.default_rng(2)
        keys = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)
        values = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3, seed=2, block_size=16)
        cache.append(mx.array(keys), mx.array(values))

        raw_nbytes = keys.nbytes + values.nbytes
        self.assertLess(cache.nbytes, raw_nbytes)

    def test_direct_cache_lazily_materializes_packed_indices(self) -> None:
        rng = np.random.default_rng(8)
        keys = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3, seed=8, block_size=4)
        cache.append(mx.array(keys), mx.array(values))

        self.assertTrue(all(block.key_packed_indices is None for block in cache._blocks))
        self.assertTrue(all(block.value_packed_indices is None for block in cache._blocks))

        state = cache.state
        self.assertGreater(len(state), 0)
        self.assertTrue(all(block.key_packed_indices is not None for block in cache._blocks))
        self.assertTrue(all(block.value_packed_indices is not None for block in cache._blocks))

    def test_mixed_bit_direct_attention_matches_dense_reconstruction(self) -> None:
        rng = np.random.default_rng(13)
        keys = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        queries = rng.standard_normal((1, 4, 5, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3.5, seed=13, block_size=2)
        query_start = cache.append(mx.array(keys), mx.array(values))
        actual = np.asarray(
            cache.direct_attention(
                mx.array(queries),
                scale=8**-0.5,
                mask="causal",
                query_start=query_start,
            )
        )

        decoded_keys, decoded_values = cache.decoded_state
        expected = dense_reference_attention(
            queries,
            np.asarray(decoded_keys),
            np.asarray(decoded_values),
            8**-0.5,
            query_start=query_start,
            causal=True,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_direct_attention_caches_rotated_blocks_for_reuse(self) -> None:
        rng = np.random.default_rng(14)
        keys = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
        queries = rng.standard_normal((1, 4, 1, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3.5, seed=14, block_size=3)
        cache.append(mx.array(keys), mx.array(values))
        self.assertTrue(all(block.key_rotated_cache is None for block in cache._blocks))
        self.assertTrue(all(block.value_rotated_cache is None for block in cache._blocks))

        cache.direct_attention(
            mx.array(queries),
            scale=8**-0.5,
            mask=None,
            query_start=5,
        )

        self.assertTrue(all(block.key_rotated_cache is not None for block in cache._blocks))
        self.assertTrue(all(block.value_rotated_cache is not None for block in cache._blocks))

    def test_lean_direct_disables_shadow_and_rotated_block_caches(self) -> None:
        rng = np.random.default_rng(15)
        keys = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
        queries = rng.standard_normal((1, 4, 1, 8)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3, seed=15, block_size=3, lean_mode=True)
        cache.append(mx.array(keys), mx.array(values))
        self.assertFalse(cache.use_index_shadow)
        self.assertFalse(cache.use_compressed_prefix_cache)
        self.assertFalse(cache.use_rotated_block_cache)
        self.assertEqual(cache.index_shadow_nbytes, 0)

        cache.direct_attention(
            mx.array(queries),
            scale=8**-0.5,
            mask=None,
            query_start=5,
        )

        self.assertEqual(cache.prefix_shadow_nbytes, 0)
        self.assertEqual(cache.rotated_block_cache_nbytes, 0)
        self.assertTrue(all(block.key_indices is None for block in cache._blocks))
        self.assertTrue(all(block.value_indices is None for block in cache._blocks))

    def test_shared_quantizer_pool_deduplicates_metadata(self) -> None:
        rng = np.random.default_rng(3)
        keys = mx.array(rng.standard_normal((1, 2, 32, 16)).astype(np.float32))
        values = mx.array(rng.standard_normal((1, 2, 32, 16)).astype(np.float32))

        independent_a = TurboQuantDirectKVCache(bits=3, seed=7, block_size=16)
        independent_b = TurboQuantDirectKVCache(bits=3, seed=7, block_size=16)
        independent_a.append(keys, values)
        independent_b.append(keys, values)

        shared_pool = TurboQuantQuantizerPool()
        shared_a = TurboQuantDirectKVCache(
            bits=3,
            seed=7,
            block_size=16,
            quantizer_pool=shared_pool,
        )
        shared_b = TurboQuantDirectKVCache(
            bits=3,
            seed=7,
            block_size=16,
            quantizer_pool=shared_pool,
        )
        shared_a.append(keys, values)
        shared_b.append(keys, values)

        independent_total = cache_list_nbytes([independent_a, independent_b])
        shared_total = cache_list_nbytes([shared_a, shared_b])
        self.assertLess(shared_total, independent_total)


if __name__ == "__main__":
    unittest.main()
