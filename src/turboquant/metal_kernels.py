from __future__ import annotations

import mlx.core as mx


def _make_turboquant_score_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        uint lane = thread_position_in_threadgroup.x;
        uint key_idx = thread_position_in_grid.y;
        uint query_idx = thread_position_in_grid.z;

        constexpr int q_per_bh = R * LQ;
        uint bh_idx = query_idx / q_per_bh;

        auto q = queries + query_idx * D;
        auto k = key_indices + (bh_idx * T + key_idx) * D;
        float norm = static_cast<float>(key_norms[bh_idx * T + key_idx]);

        float acc = 0.0f;
        constexpr int n_per_lane = (D + 31) / 32;
        for (int i = 0; i < n_per_lane; ++i) {
            uint d = lane + i * 32;
            if (d < D) {
                acc += static_cast<float>(q[d]) * static_cast<float>(centroids[static_cast<uint>(k[d])]);
            }
        }

        acc = simd_sum(acc);
        if (lane == 0) {
            scores[query_idx * T + key_idx] = static_cast<OutT>(acc * norm);
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_score_block",
        input_names=["queries", "key_indices", "key_norms", "centroids"],
        output_names=["scores"],
        source=source,
    )


_turboquant_score_kernel = _make_turboquant_score_kernel()


def has_turboquant_score_kernel() -> bool:
    return _turboquant_score_kernel is not None


def turboquant_score_block(
    rotated_queries: mx.array,
    key_indices: mx.array,
    key_norms: mx.array,
    centroids: mx.array,
) -> mx.array:
    if _turboquant_score_kernel is None:
        raise RuntimeError("TurboQuant Metal score kernel is unavailable.")

    if rotated_queries.ndim != 5:
        raise ValueError("rotated_queries must have shape [B, Hkv, R, L, D].")
    if key_indices.ndim != 4:
        raise ValueError("key_indices must have shape [B, Hkv, T, D].")
    if key_norms.ndim != 4:
        raise ValueError("key_norms must have shape [B, Hkv, T, 1].")

    B, Hkv, R, LQ, D = rotated_queries.shape
    _, _, T, key_dim = key_indices.shape
    if key_dim != D:
        raise ValueError("rotated_queries and key_indices must share the same last dimension.")

    flat_queries = mx.reshape(rotated_queries.astype(mx.float32), (B * Hkv * R * LQ, D))
    flat_indices = mx.reshape(key_indices.astype(mx.uint8), (B * Hkv * T, D))
    flat_norms = mx.reshape(key_norms.astype(mx.float32), (B * Hkv * T,))
    centroids = centroids.astype(mx.float32)

    scores = _turboquant_score_kernel(
        inputs=[flat_queries, flat_indices, flat_norms, centroids],
        template=[
            ("OutT", mx.float32),
            ("D", D),
            ("T", T),
            ("R", R),
            ("LQ", LQ),
        ],
        grid=(32, T, flat_queries.shape[0]),
        threadgroup=(32, 1, 1),
        output_shapes=[(flat_queries.shape[0], T)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.reshape(scores, (B, Hkv, R, LQ, T))
