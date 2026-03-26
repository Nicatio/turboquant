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
from turboquant.qjl import QJL


class QJLTests(unittest.TestCase):
    def test_qjl_is_empirically_unbiased(self) -> None:
        dimension = 64
        x = sample_unit_sphere(1, dimension, seed=10)[0]
        y = sample_unit_sphere(1, dimension, seed=11)[0]
        truth = float(np.dot(y, x))

        estimates = []
        for seed in range(256):
            qjl = QJL(dimension, seed=seed)
            signs = qjl.quantize_signs(x)
            estimates.append(float(np.dot(y, qjl.dequantize_signs(signs))))

        mean_estimate = float(np.mean(estimates))
        self.assertLess(abs(mean_estimate - truth), 8e-2)


if __name__ == "__main__":
    unittest.main()
