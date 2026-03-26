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

from turboquant.kv_cache import TurboQuantKVCache


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


if __name__ == "__main__":
    unittest.main()
