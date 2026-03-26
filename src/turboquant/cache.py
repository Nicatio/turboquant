from __future__ import annotations

from turboquant.lloyd_max import ScalarCodebook, build_sphere_codebook


def get_codebook(
    dimension: int,
    bits: int,
    num_grid_points: int = 16385,
    max_iter: int = 128,
    tol: float = 1e-10,
) -> ScalarCodebook:
    return build_sphere_codebook(
        dimension=dimension,
        bits=bits,
        num_grid_points=num_grid_points,
        max_iter=max_iter,
        tol=tol,
    )
