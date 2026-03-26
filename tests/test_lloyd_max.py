from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.lloyd_max import build_sphere_codebook


class LloydMaxTests(unittest.TestCase):
    def test_codebook_is_monotone(self) -> None:
        codebook = build_sphere_codebook(64, 3, num_grid_points=4097, max_iter=96)
        self.assertTrue(np.all(np.diff(codebook.centroids) > 0.0))
        self.assertTrue(np.all(np.diff(codebook.thresholds) > 0.0))

    def test_codebook_is_approximately_symmetric(self) -> None:
        codebook = build_sphere_codebook(64, 3, num_grid_points=4097, max_iter=96)
        symmetry_error = np.max(np.abs(codebook.centroids + codebook.centroids[::-1]))
        self.assertLess(symmetry_error, 2e-2)


if __name__ == "__main__":
    unittest.main()
