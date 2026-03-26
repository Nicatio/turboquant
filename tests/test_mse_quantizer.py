from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.datasets import sample_unit_sphere
from turboquant.metrics import mean_squared_error
from turboquant.mse_quantizer import TurboQuantMSE


class TurboQuantMSETests(unittest.TestCase):
    def test_quantize_and_dequantize_batch_shapes(self) -> None:
        data = sample_unit_sphere(32, 24, seed=0)
        quantizer = TurboQuantMSE(24, 2, seed=1, num_grid_points=4097, max_iter=96)
        indices = quantizer.quantize_indices(data)
        reconstruction = quantizer.dequantize_indices(indices)
        self.assertEqual(indices.shape, data.shape)
        self.assertEqual(reconstruction.shape, data.shape)

    def test_more_bits_reduce_empirical_error(self) -> None:
        data = sample_unit_sphere(128, 32, seed=2)
        low_bits = TurboQuantMSE(32, 1, seed=3, num_grid_points=4097, max_iter=96)
        high_bits = TurboQuantMSE(32, 3, seed=3, num_grid_points=4097, max_iter=96)

        low_reconstruction = low_bits.reconstruct(data)
        high_reconstruction = high_bits.reconstruct(data)

        low_error = mean_squared_error(data, low_reconstruction)
        high_error = mean_squared_error(data, high_reconstruction)
        self.assertLess(high_error, low_error)


if __name__ == "__main__":
    unittest.main()
