from __future__ import annotations

import numpy as np


def mean_squared_error(values: np.ndarray, reconstructions: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x_hat = np.asarray(reconstructions, dtype=np.float64)
    if x.shape != x_hat.shape:
        raise ValueError("values and reconstructions must have the same shape.")
    diff = x - x_hat
    if diff.ndim == 1:
        return float(np.dot(diff, diff))
    return float(np.mean(np.sum(np.square(diff), axis=1)))


def squared_inner_product_error(query: np.ndarray, value: np.ndarray, reconstruction: np.ndarray) -> float:
    y = np.asarray(query, dtype=np.float64)
    x = np.asarray(value, dtype=np.float64)
    x_hat = np.asarray(reconstruction, dtype=np.float64)
    return float((np.dot(y, x) - np.dot(y, x_hat)) ** 2)
