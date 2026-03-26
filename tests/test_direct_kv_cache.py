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

from turboquant.kv_cache import TurboQuantDirectKVCache
from turboquant.metal_kernels import has_turboquant_score_kernel, turboquant_score_block


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

    def test_direct_cache_storage_is_smaller_than_float32_cache(self) -> None:
        rng = np.random.default_rng(2)
        keys = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)
        values = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)

        cache = TurboQuantDirectKVCache(bits=3, seed=2, block_size=16)
        cache.append(mx.array(keys), mx.array(values))

        raw_nbytes = keys.nbytes + values.nbytes
        self.assertLess(cache.nbytes, raw_nbytes)


if __name__ == "__main__":
    unittest.main()
