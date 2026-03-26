from __future__ import annotations

import unittest

from turboquant.benchmark_utils import (
    contains_answer,
    exact_match,
    first_integer,
    matches_integer,
    normalize_answer,
)


class BenchmarkUtilsTest(unittest.TestCase):
    def test_normalize_answer(self) -> None:
        self.assertEqual(normalize_answer("The, Quick! Brown fox"), "quick brown fox")

    def test_exact_match(self) -> None:
        self.assertTrue(exact_match("Blue Sparrow", ["blue sparrow"]))
        self.assertFalse(exact_match("Blue Sparrow today", ["blue sparrow"]))

    def test_contains_answer(self) -> None:
        self.assertTrue(contains_answer("The answer is blue sparrow.", ["blue sparrow"]))
        self.assertFalse(contains_answer("The answer is green sparrow.", ["blue sparrow"]))

    def test_integer_matching(self) -> None:
        self.assertEqual(first_integer("Paragraph 17 contains the answer."), 17)
        self.assertTrue(matches_integer("17", ["Paragraph 17"]))
        self.assertFalse(matches_integer("18", ["Paragraph 17"]))


if __name__ == "__main__":
    unittest.main()
