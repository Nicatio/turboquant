from __future__ import annotations

import argparse

import numpy as np

from turboquant.datasets import sample_unit_sphere
from turboquant.nn_eval import recall_at_k
from turboquant.prod_quantizer import TurboQuantProd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small synthetic nearest-neighbor benchmark.")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--database-size", type=int, default=2000)
    parser.add_argument("--query-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    database = sample_unit_sphere(args.database_size, args.dimension, rng=rng)
    queries = sample_unit_sphere(args.query_size, args.dimension, rng=rng)

    quantizer = TurboQuantProd(args.dimension, args.bits, seed=args.seed, num_grid_points=8193, max_iter=96)
    approx_database = np.vstack([quantizer.reconstruct(vector) for vector in database])

    recall = recall_at_k(queries, database, approx_database, args.k)
    print(f"recall@{args.k}={recall:.6f}")


if __name__ == "__main__":
    main()
