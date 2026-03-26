from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.kv_cache import TurboQuantKVCache


class TurboQuantKVCacheTests(unittest.TestCase):
    def test_update_and_fetch_preserves_shape(self) -> None:
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
        values = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)

        cache = TurboQuantKVCache(bits=3, seed=0)
        returned_keys, returned_values = cache.update_and_fetch(keys, values)

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
        cache.update_and_fetch(keys, values)

        raw_nbytes = keys.nbytes + values.nbytes
        self.assertLess(cache.nbytes, raw_nbytes)


if __name__ == "__main__":
    unittest.main()
