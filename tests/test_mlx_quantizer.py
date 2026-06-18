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

from turboquant.metal_kernels import has_turboquant_quantize_kernel, turboquant_quantize_rotated
from turboquant.mse_quantizer import TurboQuantMSE
from turboquant.mlx_quantizer import (
    MixedPackedIndexCodec,
    MlxMixedBitTurboQuantMSE,
    MlxTurboQuantMSE,
)


class MlxTurboQuantMSETests(unittest.TestCase):
    def test_quantize_kernel_matches_threshold_count(self) -> None:
        if not has_turboquant_quantize_kernel():
            self.skipTest("TurboQuant Metal quantize kernel is unavailable.")

        quantizer = TurboQuantMSE(
            dimension=8,
            bits=3,
            seed=11,
            require_unit_norm=True,
        )
        mlx_quantizer = MlxTurboQuantMSE(quantizer)
        rng = np.random.default_rng(11)
        values = mx.array(rng.standard_normal((5, 8)).astype(np.float32))
        rotated = mlx_quantizer.rotate(values)

        actual = np.asarray(
            turboquant_quantize_rotated(mx.reshape(rotated, (-1, 8)), mlx_quantizer.thresholds)
        )
        expected = np.sum(
            np.asarray(rotated)[..., None] > np.asarray(mlx_quantizer.thresholds),
            axis=-1,
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_mixed_codec_round_trip_preserves_indices(self) -> None:
        rng = np.random.default_rng(23)
        high = rng.integers(0, 16, size=(2, 3, 5, 4), dtype=np.uint8)
        low = rng.integers(0, 8, size=(2, 3, 5, 4), dtype=np.uint8)
        indices = mx.array(np.concatenate([high, low], axis=-1))
        codec = MixedPackedIndexCodec(low_bits=3, high_bits=4, high_dims=4)

        packed = codec.pack(indices)
        unpacked = codec.unpack(packed, tuple(indices.shape))

        np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(indices))

    def test_mixed_bit_quantizer_preserves_shape_and_bounds(self) -> None:
        rng = np.random.default_rng(31)
        quantizer = MlxMixedBitTurboQuantMSE(
            dimension=8,
            bits=3.5,
            seed=31,
        )
        values = mx.array(rng.standard_normal((6, 8)).astype(np.float32))

        indices = quantizer.quantize_indices(values)
        reconstructed = quantizer.dequantize_indices(indices)

        self.assertEqual(tuple(indices.shape), (6, 8))
        self.assertEqual(tuple(reconstructed.shape), (6, 8))
        self.assertLess(int(np.asarray(indices[:, : quantizer.high_dims]).max()), 1 << quantizer.high_bits)
        self.assertLess(int(np.asarray(indices[:, quantizer.high_dims :]).max()), 1 << quantizer.low_bits)


if __name__ == "__main__":
    unittest.main()
