from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from turboquant.benchmark_utils import bytes_to_gb
from turboquant.datasets import normalize_rows
from turboquant.nn_eval import recall_at_k
from turboquant.prod_quantizer import TurboQuantProd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_glove_subset(
    dataset_path: Path,
    database_size: int,
    query_size: int,
    seed: int,
):
    generator = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as handle:
        available_train = handle["train"].shape[0]
        available_test = handle["test"].shape[0]
        database_size = min(database_size, available_train)
        query_size = min(query_size, available_test)
        database_indices = generator.choice(available_train, size=database_size, replace=False)
        query_indices = generator.choice(available_test, size=query_size, replace=False)
        database = np.asarray(handle["train"][np.sort(database_indices)], dtype=np.float64)
        queries = np.asarray(handle["test"][np.sort(query_indices)], dtype=np.float64)
    return normalize_rows(database, axis=1), normalize_rows(queries, axis=1)


def estimate_encoded_storage_bytes(
    num_vectors: int,
    dimension: int,
    bits: int,
    residual_norm_bytes: int = 2,
) -> int:
    packed_bits = num_vectors * dimension * bits
    packed_bytes = int(math.ceil(packed_bits / 8.0))
    residual_bytes = num_vectors * residual_norm_bytes
    return packed_bytes + residual_bytes


def quantizer_overhead_bytes(quantizer: TurboQuantProd) -> int:
    total = quantizer.qjl.projection.nbytes
    if quantizer.mse_quantizer is not None:
        total += quantizer.mse_quantizer.rotation.matrix.nbytes
        total += quantizer.mse_quantizer.codebook.centroids.nbytes
        total += quantizer.mse_quantizer.codebook.thresholds.nbytes
    return int(total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a public GloVe nearest-neighbor benchmark for TurboQuant recall and storage."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "glove-100-angular.hdf5",
        help="Path to glove-100-angular.hdf5.",
    )
    parser.add_argument("--database-size", type=int, default=20000)
    parser.add_argument("--query-size", type=int, default=512)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}. Download glove-100-angular.hdf5 first."
        )

    database, queries = load_glove_subset(
        dataset_path=args.dataset,
        database_size=args.database_size,
        query_size=args.query_size,
        seed=args.seed,
    )
    dimension = database.shape[1]

    quantizer = TurboQuantProd(
        dimension=dimension,
        bits=args.bits,
        seed=args.seed,
        num_grid_points=8193,
        max_iter=96,
        require_unit_norm=True,
    )
    approx_database = np.vstack([quantizer.reconstruct(vector) for vector in database])

    baseline_bytes = int(database.astype(np.float32).nbytes)
    encoded_bytes = estimate_encoded_storage_bytes(
        num_vectors=database.shape[0],
        dimension=dimension,
        bits=args.bits,
    ) + quantizer_overhead_bytes(quantizer)
    recall = recall_at_k(queries, database, approx_database, args.k)
    mean_cosine = float(np.mean(np.sum(database * approx_database, axis=1)))
    mean_mse = float(np.mean(np.sum((database - approx_database) ** 2, axis=1)))

    print(f"dataset={args.dataset}")
    print(f"database_size={database.shape[0]}")
    print(f"query_size={queries.shape[0]}")
    print(f"dimension={dimension}")
    print(f"bits={args.bits}")
    print(f"recall@{args.k}={recall:.6f}")
    print(f"mean_vector_cosine={mean_cosine:.6f}")
    print(f"mean_vector_mse={mean_mse:.6f}")
    print(f"baseline_storage_gb={bytes_to_gb(baseline_bytes):.6f}")
    print(f"turboquant_storage_gb={bytes_to_gb(encoded_bytes):.6f}")
    print(f"storage_compression_ratio={baseline_bytes / max(encoded_bytes, 1):.4f}")


if __name__ == "__main__":
    main()
