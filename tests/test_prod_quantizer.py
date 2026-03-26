from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.datasets import sample_unit_sphere
from turboquant.prod_quantizer import TurboQuantProd


class TurboQuantProdTests(unittest.TestCase):
    def test_bits_one_path_runs(self) -> None:
        vector = sample_unit_sphere(1, 16, seed=0)[0]
        quantizer = TurboQuantProd(16, 1, seed=1, num_grid_points=2049, max_iter=64)
        encoded = quantizer.quantize(vector)
        reconstruction = quantizer.dequantize(encoded)
        self.assertEqual(reconstruction.shape, vector.shape)

    def test_prod_estimator_is_empirically_near_unbiased(self) -> None:
        dimension = 32
        x = sample_unit_sphere(1, dimension, seed=20)[0]
        y = sample_unit_sphere(1, dimension, seed=21)[0]
        truth = float(np.dot(y, x))

        estimates = []
        for seed in range(160):
            quantizer = TurboQuantProd(dimension, 3, seed=seed, num_grid_points=4097, max_iter=96)
            encoded = quantizer.quantize(x)
            estimates.append(quantizer.estimate_inner_product(y, encoded))

        mean_estimate = float(np.mean(estimates))
        self.assertLess(abs(mean_estimate - truth), 1.2e-1)


if __name__ == "__main__":
    unittest.main()
