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
from turboquant.rotation import generate_random_rotation


class RotationTests(unittest.TestCase):
    def test_rotation_is_orthogonal(self) -> None:
        rotation = generate_random_rotation(16, seed=0)
        identity = np.einsum("ki,kj->ij", rotation.matrix, rotation.matrix)
        self.assertTrue(np.allclose(identity, np.eye(16), atol=1e-10))

    def test_apply_then_inverse_recovers_vector(self) -> None:
        vector = sample_unit_sphere(1, 16, seed=1)[0]
        rotation = generate_random_rotation(16, seed=2)
        recovered = rotation.inverse(rotation.apply(vector))
        self.assertTrue(np.allclose(recovered, vector, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
