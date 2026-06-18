# TurboQuant Math Mapping

This note records the formulas implemented in the first local reference version.

## Coordinate Distribution After Rotation

For a unit vector `x in S^(d-1)` and a random orthogonal rotation `Pi`, each coordinate of `Pi x`
has the same marginal density as the first coordinate of a uniformly random point on the sphere:

`f_d(t) = C_d * (1 - t^2)^((d - 3) / 2)` for `t in (-1, 1)`

with

`C_d = Gamma(d / 2) / (sqrt(pi) * Gamma((d - 1) / 2))`

The code uses this density to build a discrete grid approximation and then solves a 1D Lloyd-Max
problem on that grid.

## TurboQuantMSE Reference Pipeline

Given a unit vector `x`:

1. Draw a random orthogonal matrix `Pi`.
2. Compute `u = Pi x`.
3. Quantize each coordinate of `u` with the same scalar codebook:
   `idx_j = QuantScalar(u_j)`.
4. Reconstruct `u_hat_j = c[idx_j]` from the Lloyd-Max centroids.
5. Return `x_hat = Pi^T u_hat`.

This is the paper's MSE-focused path, implemented as a CPU-first reference version.

## QJL Residual Correction

For the inner-product estimator, the implementation follows Algorithm 2:

1. Build `x_hat_mse` with a `b - 1` bit `TurboQuantMSE`.
2. Compute the residual `r = x - x_hat_mse`.
3. Draw a Gaussian matrix `S`.
4. Store `sign(S r)` and `gamma = ||r||_2`.
5. Reconstruct the residual estimate as
   `r_hat = sqrt(pi / 2) / d * gamma * S^T sign(S r)`.
6. Output `x_hat = x_hat_mse + r_hat`.

For a fixed MSE reconstruction, the QJL term is unbiased in expectation over the random Gaussian
projection matrix.

## Intentional Simplifications In This Version

- Dense QR rotations are used instead of faster structured transforms.
- Lloyd-Max centroids are solved numerically on a discrete grid instead of with closed-form
  integrations.
- Experiments target local CPU validation, not the largest paper-scale GPU experiments.
