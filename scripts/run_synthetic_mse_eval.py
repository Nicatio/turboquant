from __future__ import annotations

import argparse

import numpy as np

from turboquant.datasets import sample_unit_sphere
from turboquant.metrics import mean_squared_error
from turboquant.mse_quantizer import TurboQuantMSE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic TurboQuant MSE benchmark.")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = sample_unit_sphere(args.samples, args.dimension, seed=args.seed)
    print("bits\tavg_squared_l2_error")
    for bits in range(1, args.max_bits + 1):
        quantizer = TurboQuantMSE(args.dimension, bits, seed=args.seed, num_grid_points=8193, max_iter=96)
        reconstruction = quantizer.reconstruct(data)
        error = mean_squared_error(data, reconstruction)
        print(f"{bits}\t{error:.6f}")


if __name__ == "__main__":
    main()
