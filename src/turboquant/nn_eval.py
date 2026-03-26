from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def topk_inner_product_indices(query: np.ndarray, database: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive.")
    scores = np.einsum(
        "nd,d->n",
        np.asarray(database, dtype=np.float64),
        np.asarray(query, dtype=np.float64),
    )
    k = min(k, scores.shape[0])
    partition = np.argpartition(-scores, kth=k - 1)[:k]
    return partition[np.argsort(-scores[partition])]


def recall_at_k(queries: np.ndarray, database: np.ndarray, approx_database: np.ndarray, k: int) -> float:
    exact_queries = np.asarray(queries, dtype=np.float64)
    exact_database = np.asarray(database, dtype=np.float64)
    approx = np.asarray(approx_database, dtype=np.float64)
    if exact_database.shape != approx.shape:
        raise ValueError("database and approx_database must have the same shape.")

    recalls = []
    for query in exact_queries:
        exact_topk = set(topk_inner_product_indices(query, exact_database, k))
        approx_topk = set(topk_inner_product_indices(query, approx, k))
        recalls.append(len(exact_topk & approx_topk) / float(len(exact_topk)))
    return float(np.mean(recalls))
