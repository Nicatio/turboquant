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
from turboquant.mlx_attention import TurboQuantQwen3_5Attention


class _Identity:
    def __call__(self, x):
        return x


class _FakeQProj:
    def __call__(self, x):
        return mx.concatenate([x, mx.zeros_like(x)], axis=-1)


class _FakeRotary:
    def __call__(self, x, position_ids):
        batch = position_ids.shape[1]
        seq = position_ids.shape[2]
        dim = x.shape[-1]
        return (
            mx.ones((batch, seq, dim), dtype=x.dtype),
            mx.zeros((batch, seq, dim), dtype=x.dtype),
        )


class _FakeBaseQwenAttention:
    num_key_value_heads = 1
    num_attention_heads = 1
    head_dim = 2
    scale = 2**-0.5
    q_proj = _FakeQProj()
    k_proj = _Identity()
    v_proj = _Identity()
    o_proj = _Identity()
    q_norm = _Identity()
    k_norm = _Identity()
    rotary_emb = _FakeRotary()


class _FakeDirectCache(TurboQuantDirectKVCache):
    def __init__(self):
        self.offset = 4
        self.appended = None
        self.called = None

    def append(self, keys: mx.array, values: mx.array) -> int:
        query_start = self.offset
        self.appended = (keys, values)
        self.offset += keys.shape[2]
        return query_start

    def direct_attention(self, queries: mx.array, *, scale: float, mask=None, query_start: int):
        self.called = {
            "scale": scale,
            "mask": mask,
            "query_start": query_start,
            "shape": tuple(queries.shape),
        }
        return mx.ones(queries.shape, dtype=queries.dtype)


class TurboQuantQwenAttentionTests(unittest.TestCase):
    def test_qwen_wrapper_uses_direct_attention_cache(self) -> None:
        attention = TurboQuantQwen3_5Attention(_FakeBaseQwenAttention())
        cache = _FakeDirectCache()
        x = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
        position_ids = mx.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=mx.int32)

        actual = np.asarray(attention(x, mask="causal", cache=cache, position_ids=position_ids))

        expected = np.full((1, 2, 2), 0.5, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertIsNotNone(cache.appended)
        self.assertEqual(cache.called["query_start"], 4)
        self.assertEqual(cache.called["mask"], "causal")
        self.assertEqual(cache.called["shape"], (1, 1, 2, 2))
        self.assertEqual(cache.offset, 6)


if __name__ == "__main__":
    unittest.main()
