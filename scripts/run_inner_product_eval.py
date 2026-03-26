from __future__ import annotations

import argparse

import numpy as np

from turboquant.datasets import sample_unit_sphere
from turboquant.prod_quantizer import TurboQuantProd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TurboQuantProd inner-product bias locally.")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_rng = np.random.default_rng(args.seed)
    x = sample_unit_sphere(1, args.dimension, rng=base_rng)[0]
    y = sample_unit_sphere(1, args.dimension, rng=base_rng)[0]
    truth = float(np.dot(y, x))

    estimates = []
    for trial in range(args.trials):
        quantizer = TurboQuantProd(
            dimension=args.dimension,
            bits=args.bits,
            seed=args.seed + trial,
            num_grid_points=8193,
            max_iter=96,
        )
        encoded = quantizer.quantize(x)
        estimates.append(quantizer.estimate_inner_product(y, encoded))

    estimates_array = np.asarray(estimates, dtype=np.float64)
    print(f"truth={truth:.6f}")
    print(f"mean_estimate={np.mean(estimates_array):.6f}")
    print(f"bias={np.mean(estimates_array) - truth:.6f}")
    print(f"std={np.std(estimates_array):.6f}")


if __name__ == "__main__":
    main()
