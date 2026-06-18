from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.kv_cache import TurboQuantDirectKVCache
from turboquant.mlx_attention import (
    TurboQuantGemma4Attention,
    TurboQuantQwen3_5Attention,
    TurboQuantQwen3VLAttention,
)


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
    n_heads = 1
    n_kv_heads = 1
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


class _FakeBaseQwenVLAttention:
    n_heads = 1
    n_kv_heads = 1
    head_dim = 2
    scale = 2**-0.5
    q_proj = _Identity()
    k_proj = _Identity()
    v_proj = _Identity()
    o_proj = _Identity()
    q_norm = _Identity()
    k_norm = _Identity()
    rotary_emb = _FakeRotary()


class _FakeBaseGemmaAttention:
    head_dim = 2
    n_heads = 1
    n_kv_heads = 1
    scale = 2**-0.5
    use_k_eq_v = False
    is_sliding = False
    q_proj = _Identity()
    k_proj = _Identity()
    v_proj = _Identity()
    o_proj = _Identity()
    q_norm = _Identity()
    k_norm = _Identity()
    v_norm = _Identity()

    class _Config:
        sliding_window = 4

    config = _Config()

    class _Rope:
        def __call__(self, x, offset=0):
            return x

    rope = _Rope()


class _FakeDirectCache(TurboQuantDirectKVCache):
    def __init__(self):
        self.offset = 4
        self.appended = None
        self.called = None
        self._decoded_state = (
            mx.ones((1, 1, 2, 2), dtype=mx.float32),
            mx.ones((1, 1, 2, 2), dtype=mx.float32),
        )

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

    @property
    def decoded_state(self):
        return self._decoded_state


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

    def test_qwen3_vl_wrapper_uses_direct_attention_cache(self) -> None:
        attention = TurboQuantQwen3VLAttention(_FakeBaseQwenVLAttention())
        cache = _FakeDirectCache()
        x = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
        position_ids = mx.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=mx.int32)

        actual = np.asarray(attention(x, mask="causal", cache=cache, position_ids=position_ids))

        expected = np.ones((1, 2, 2), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertIsNotNone(cache.appended)
        self.assertEqual(cache.called["query_start"], 4)
        self.assertEqual(cache.called["mask"], "causal")
        self.assertEqual(cache.called["shape"], (1, 1, 2, 2))
        self.assertEqual(cache.offset, 6)

    def test_gemma4_wrapper_uses_direct_attention_cache(self) -> None:
        attention = TurboQuantGemma4Attention(_FakeBaseGemmaAttention())
        cache = _FakeDirectCache()
        x = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)

        actual, shared_kv, rope_offset = attention(x, mask="causal", cache=cache)
        actual = np.asarray(actual)

        expected = np.ones((1, 2, 2), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertIsNotNone(cache.appended)
        self.assertEqual(cache.called["query_start"], 4)
        self.assertEqual(cache.called["mask"], "causal")
        self.assertEqual(cache.called["shape"], (1, 1, 2, 2))
        self.assertEqual(tuple(shared_kv[0].shape), (1, 1, 2, 2))
        self.assertEqual(cache.offset, 6)
        self.assertEqual(int(rope_offset), 4)

    def test_gemma4_wrapper_rebuilds_sliding_mask_for_shared_kv(self) -> None:
        class _SlidingGemmaAttention(_FakeBaseGemmaAttention):
            is_sliding = True

        attention = TurboQuantGemma4Attention(_SlidingGemmaAttention())
        x = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
        shared_keys = mx.ones((1, 1, 5, 2), dtype=mx.float32)
        shared_values = mx.ones((1, 1, 5, 2), dtype=mx.float32)
        short_mask = mx.ones((2, 2), dtype=mx.bool_)

        def _fake_sdpa(queries, keys, values, cache, scale, mask):
            self.assertEqual(tuple(queries.shape), (1, 1, 2, 2))
            self.assertEqual(tuple(keys.shape), (1, 1, 5, 2))
            self.assertEqual(tuple(values.shape), (1, 1, 5, 2))
            self.assertEqual(tuple(mask.shape), (2, 5))
            return mx.ones_like(queries)

        with patch("mlx_lm.models.base.scaled_dot_product_attention", _fake_sdpa):
            actual, shared_kv, rope_offset = attention(
                x,
                mask=short_mask,
                cache=None,
                shared_kv=(shared_keys, shared_values),
                offset=mx.array(3),
            )

        np.testing.assert_allclose(np.asarray(actual), np.ones((1, 2, 2), dtype=np.float32))
        self.assertIs(shared_kv[0], shared_keys)
        self.assertIs(shared_kv[1], shared_values)
        self.assertEqual(int(rope_offset), 3)


if __name__ == "__main__":
    unittest.main()
