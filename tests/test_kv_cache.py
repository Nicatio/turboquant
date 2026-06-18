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

from turboquant.kv_cache import TurboQuantKVCache, cache_packed_nbytes


class TurboQuantKVCacheTests(unittest.TestCase):
    def test_packed_storage_round_trip_preserves_shape(self) -> None:
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3, seed=0)
        cache.update_and_fetch(mx.array(keys), mx.array(values))
        decoded_keys, decoded_values = cache.state

        self.assertEqual(np.asarray(decoded_keys).shape, keys.shape)
        self.assertEqual(np.asarray(decoded_values).shape, values.shape)

    def test_update_and_fetch_preserves_shape(self) -> None:
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3, seed=0)
        returned_keys, returned_values = cache.update_and_fetch(mx.array(keys), mx.array(values))

        returned_keys_np = np.asarray(returned_keys)
        returned_values_np = np.asarray(returned_values)

        self.assertEqual(returned_keys_np.shape, keys.shape)
        self.assertEqual(returned_values_np.shape, values.shape)
        self.assertGreater(cache.nbytes, 0)

    def test_storage_is_smaller_than_float32_cache(self) -> None:
        rng = np.random.default_rng(1)
        keys = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)
        values = rng.standard_normal((1, 4, 64, 16)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3, seed=1)
        cache.update_and_fetch(mx.array(keys), mx.array(values))

        raw_nbytes = keys.nbytes + values.nbytes
        self.assertLess(cache.nbytes, raw_nbytes)

    def test_state_round_trip_exposes_decoded_arrays(self) -> None:
        rng = np.random.default_rng(2)
        keys = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3, seed=2)
        cache.update_and_fetch(mx.array(keys), mx.array(values))
        decoded_keys, decoded_values = cache.state

        self.assertEqual(np.asarray(decoded_keys).shape, keys.shape)
        self.assertEqual(np.asarray(decoded_values).shape, values.shape)

    def test_dense_shadow_path_preserves_shape(self) -> None:
        rng = np.random.default_rng(3)
        keys = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 6, 8)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3, seed=3, use_dense_shadow=True)
        returned_keys, returned_values = cache.update_and_fetch(mx.array(keys), mx.array(values))

        self.assertEqual(np.asarray(returned_keys).shape, keys.shape)
        self.assertEqual(np.asarray(returned_values).shape, values.shape)
        self.assertGreater(cache.dense_nbytes, 0)
        self.assertGreater(cache.index_shadow_nbytes, 0)
        self.assertTrue(all(chunk.packed_indices is None for chunk in cache._key_chunks + cache._value_chunks))

    def test_dense_shadow_recent_window_keeps_recent_tokens_dense(self) -> None:
        rng = np.random.default_rng(4)
        keys = rng.standard_normal((1, 2, 7, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 7, 8)).astype(np.float32)

        full_cache = TurboQuantKVCache(bits=3, seed=4, use_dense_shadow=True)
        full_keys, full_values = full_cache.update_and_fetch(mx.array(keys), mx.array(values))

        recent_cache = TurboQuantKVCache(
            bits=3,
            seed=4,
            use_dense_shadow=True,
            recent_window_tokens=3,
        )
        recent_keys, recent_values = recent_cache.update_and_fetch(mx.array(keys), mx.array(values))

        full_keys_np = np.asarray(full_keys)
        full_values_np = np.asarray(full_values)
        recent_keys_np = np.asarray(recent_keys)
        recent_values_np = np.asarray(recent_values)

        np.testing.assert_allclose(recent_keys_np[..., :4, :], full_keys_np[..., :4, :], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            recent_values_np[..., :4, :], full_values_np[..., :4, :], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(recent_keys_np[..., 4:, :], keys[..., 4:, :], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            recent_values_np[..., 4:, :], values[..., 4:, :], rtol=1e-6, atol=1e-6
        )
        self.assertLess(cache_packed_nbytes(recent_cache), keys.nbytes + values.nbytes)

    def test_dense_shadow_recent_slack_defers_single_token_flushes(self) -> None:
        rng = np.random.default_rng(5)
        prefix_keys = mx.array(rng.standard_normal((1, 2, 4, 8)).astype(np.float32))
        prefix_values = mx.array(rng.standard_normal((1, 2, 4, 8)).astype(np.float32))
        token_keys = mx.array(rng.standard_normal((1, 2, 1, 8)).astype(np.float32))
        token_values = mx.array(rng.standard_normal((1, 2, 1, 8)).astype(np.float32))

        cache = TurboQuantKVCache(
            bits=3,
            seed=5,
            use_dense_shadow=True,
            recent_window_tokens=2,
            recent_slack_tokens=2,
        )
        cache.update_and_fetch(prefix_keys, prefix_values)
        self.assertEqual(len(cache._key_chunks), 1)
        self.assertEqual(cache._recent_keys.shape[2], 2)

        cache.update_and_fetch(token_keys, token_values)
        self.assertEqual(len(cache._key_chunks), 1)
        self.assertEqual(cache._recent_keys.shape[2], 3)

        cache.update_and_fetch(token_keys, token_values)
        self.assertEqual(len(cache._key_chunks), 1)
        self.assertEqual(cache._recent_keys.shape[2], 4)

        cache.update_and_fetch(token_keys, token_values)
        self.assertEqual(len(cache._key_chunks), 2)
        self.assertEqual(cache._recent_keys.shape[2], 4)

    def test_mixed_bit_shadow_cache_round_trip_preserves_shape(self) -> None:
        rng = np.random.default_rng(6)
        keys = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 5, 8)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3.5, seed=6)
        cache.update_and_fetch(mx.array(keys), mx.array(values))
        decoded_keys, decoded_values = cache.state

        self.assertEqual(np.asarray(decoded_keys).shape, keys.shape)
        self.assertEqual(np.asarray(decoded_values).shape, values.shape)


if __name__ == "__main__":
    unittest.main()
