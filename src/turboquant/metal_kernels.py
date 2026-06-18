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


def _make_turboquant_value_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        uint d = thread_position_in_grid.x;
        uint query_idx = thread_position_in_grid.y;

        constexpr int q_per_bh = R * LQ;
        uint bh_idx = query_idx / q_per_bh;

        auto w = weights + query_idx * T;
        auto v = value_indices + (bh_idx * T * D) + d;
        auto n = value_norms + bh_idx * T;

        float acc = 0.0f;
        for (uint t = 0; t < T; ++t) {
            float weight = static_cast<float>(w[t]);
            float norm = static_cast<float>(n[t]);
            uint idx = static_cast<uint>(v[t * D]);
            acc += weight * static_cast<float>(centroids[idx]) * norm;
        }

        output[query_idx * D + d] = static_cast<OutT>(acc);
    """
    return mx.fast.metal_kernel(
        name="turboquant_value_block",
        input_names=["weights", "value_indices", "value_norms", "centroids"],
        output_names=["output"],
        source=source,
    )


_turboquant_value_kernel = _make_turboquant_value_kernel()


def _make_turboquant_quantize_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        uint d = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;

        float value = static_cast<float>(rotated[row * D + d]);
        uint idx = 0;
        for (uint i = 0; i < K; ++i) {
            idx += value > static_cast<float>(thresholds[i]) ? 1 : 0;
        }
        indices[row * D + d] = static_cast<OutT>(idx);
    """
    return mx.fast.metal_kernel(
        name="turboquant_quantize_rotated",
        input_names=["rotated", "thresholds"],
        output_names=["indices"],
        source=source,
    )


_turboquant_quantize_kernel = _make_turboquant_quantize_kernel()


def _make_turboquant_fused_attention_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        uint lane = thread_position_in_threadgroup.x;
        uint query_idx = thread_position_in_grid.y;

        uint q_per_bh = R * LQ;
        uint bh_idx = query_idx / q_per_bh;
        auto q = queries + query_idx * KD;

        threadgroup float scores[T];
        threadgroup float reductions[2];

        for (uint t = 0; t < T; ++t) {
            auto k = key_indices + (bh_idx * T + t) * KD;
            float norm = static_cast<float>(key_norms[bh_idx * T + t]);

            float acc = 0.0f;
            constexpr int n_per_lane = (KD + 31) / 32;
            for (int i = 0; i < n_per_lane; ++i) {
                uint d = lane + i * 32;
                if (d < KD) {
                    acc += static_cast<float>(q[d]) *
                           static_cast<float>(key_centroids[static_cast<uint>(k[d])]);
                }
            }

            acc = simd_sum(acc);
            if (lane == 0) {
                bool masked = false;
                if (Causal) {
                    masked = static_cast<int>(query_positions[query_idx]) <
                             static_cast<int>(key_positions[t]);
                }
                scores[t] = masked ? -INFINITY : (acc * norm);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            float max_score = -INFINITY;
            for (uint t = 0; t < T; ++t) {
                max_score = max(max_score, scores[t]);
            }

            float denom = 0.0f;
            for (uint t = 0; t < T; ++t) {
                float weight = exp(scores[t] - max_score);
                scores[t] = weight;
                denom += weight;
            }

            reductions[0] = max_score;
            reductions[1] = denom;
            block_max[query_idx] = static_cast<OutT>(max_score);
            block_sum[query_idx] = static_cast<OutT>(denom);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lane; d < VD; d += 32) {
            float acc = 0.0f;
            auto v = value_indices + (bh_idx * T * VD) + d;
            auto n = value_norms + bh_idx * T;
            for (uint t = 0; t < T; ++t) {
                float norm = static_cast<float>(n[t]);
                uint idx = static_cast<uint>(v[t * VD]);
                acc += scores[t] * static_cast<float>(value_centroids[idx]) * norm;
            }
            block_output[query_idx * VD + d] = static_cast<OutT>(acc);
        }
    """
    return mx.fast.metal_kernel(
        name="turboquant_fused_block_attention",
        input_names=[
            "queries",
            "key_indices",
            "key_norms",
            "key_centroids",
            "query_positions",
            "key_positions",
            "value_indices",
            "value_norms",
            "value_centroids",
        ],
        output_names=["block_output", "block_max", "block_sum"],
        source=source,
    )


_turboquant_fused_attention_kernel = _make_turboquant_fused_attention_kernel()


def has_turboquant_score_kernel() -> bool:
    return _turboquant_score_kernel is not None


def has_turboquant_value_kernel() -> bool:
    return _turboquant_value_kernel is not None


def has_turboquant_quantize_kernel() -> bool:
    return _turboquant_quantize_kernel is not None


def has_turboquant_fused_attention_kernel() -> bool:
    return _turboquant_fused_attention_kernel is not None


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


def turboquant_value_block(
    weights: mx.array,
    value_indices: mx.array,
    value_norms: mx.array,
    centroids: mx.array,
) -> mx.array:
    if _turboquant_value_kernel is None:
        raise RuntimeError("TurboQuant Metal value kernel is unavailable.")

    if weights.ndim != 5:
        raise ValueError("weights must have shape [B, Hkv, R, L, T].")
    if value_indices.ndim != 4:
        raise ValueError("value_indices must have shape [B, Hkv, T, D].")
    if value_norms.ndim != 4:
        raise ValueError("value_norms must have shape [B, Hkv, T, 1].")

    B, Hkv, R, LQ, T = weights.shape
    _, _, value_tokens, D = value_indices.shape
    if value_tokens != T:
        raise ValueError("weights and value_indices must agree on the token dimension.")

    flat_weights = mx.reshape(weights.astype(mx.float32), (B * Hkv * R * LQ, T))
    flat_indices = mx.reshape(value_indices.astype(mx.uint8), (B * Hkv * T, D))
    flat_norms = mx.reshape(value_norms.astype(mx.float32), (B * Hkv * T,))
    centroids = centroids.astype(mx.float32)

    output = _turboquant_value_kernel(
        inputs=[flat_weights, flat_indices, flat_norms, centroids],
        template=[
            ("OutT", mx.float32),
            ("D", D),
            ("T", T),
            ("R", R),
            ("LQ", LQ),
        ],
        grid=(D, flat_weights.shape[0], 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(flat_weights.shape[0], D)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.reshape(output, (B, Hkv, R, LQ, D))


def turboquant_quantize_rotated(
    rotated: mx.array,
    thresholds: mx.array,
) -> mx.array:
    if _turboquant_quantize_kernel is None:
        raise RuntimeError("TurboQuant Metal quantize kernel is unavailable.")

    if rotated.ndim != 2:
        raise ValueError("rotated must have shape [N, D].")
    if thresholds.ndim != 1:
        raise ValueError("thresholds must have shape [K].")

    rows, dimension = rotated.shape
    num_thresholds = thresholds.shape[0]
    if num_thresholds == 0:
        return mx.zeros(rotated.shape, dtype=mx.uint8)

    return _turboquant_quantize_kernel(
        inputs=[rotated.astype(mx.float32), thresholds.astype(mx.float32)],
        template=[
            ("OutT", mx.uint8),
            ("D", dimension),
            ("K", num_thresholds),
        ],
        grid=(dimension, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, dimension)],
        output_dtypes=[mx.uint8],
    )[0]


def turboquant_fused_block_attention(
    rotated_queries: mx.array,
    key_indices: mx.array,
    key_norms: mx.array,
    key_centroids: mx.array,
    query_positions: mx.array | None,
    key_positions: mx.array | None,
    value_indices: mx.array,
    value_norms: mx.array,
    value_centroids: mx.array,
    *,
    causal: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    if _turboquant_fused_attention_kernel is None:
        raise RuntimeError("TurboQuant Metal fused attention kernel is unavailable.")

    if rotated_queries.ndim not in (4, 5):
        raise ValueError("rotated_queries must have shape [B, Hkv, R, KD] or [B, Hkv, R, L, KD].")
    if key_indices.ndim != 4:
        raise ValueError("key_indices must have shape [B, Hkv, T, KD].")
    if key_norms.ndim != 4:
        raise ValueError("key_norms must have shape [B, Hkv, T, 1].")
    if value_indices.ndim != 4:
        raise ValueError("value_indices must have shape [B, Hkv, T, VD].")
    if value_norms.ndim != 4:
        raise ValueError("value_norms must have shape [B, Hkv, T, 1].")

    if rotated_queries.ndim == 4:
        rotated_queries = mx.expand_dims(rotated_queries, axis=3)
    B, Hkv, R, LQ, KD = rotated_queries.shape
    _, _, T, key_dim = key_indices.shape
    _, _, value_tokens, VD = value_indices.shape
    if key_dim != KD:
        raise ValueError("rotated_queries and key_indices must share the same key dimension.")
    if value_tokens != T:
        raise ValueError("key_indices and value_indices must agree on the token dimension.")
    if causal:
        if query_positions is None or key_positions is None:
            raise ValueError("causal fused attention requires query_positions and key_positions.")
    else:
        if query_positions is None:
            query_positions = mx.zeros((B, Hkv, R, LQ), dtype=mx.int32)
        if key_positions is None:
            key_positions = mx.zeros((T,), dtype=mx.int32)

    query_count = B * Hkv * R * LQ
    flat_queries = mx.reshape(rotated_queries.astype(mx.float32), (query_count, KD))
    flat_key_indices = mx.reshape(key_indices.astype(mx.uint8), (B * Hkv * T, KD))
    flat_key_norms = mx.reshape(key_norms.astype(mx.float32), (B * Hkv * T,))
    flat_value_indices = mx.reshape(value_indices.astype(mx.uint8), (B * Hkv * T, VD))
    flat_value_norms = mx.reshape(value_norms.astype(mx.float32), (B * Hkv * T,))
    flat_query_positions = mx.reshape(query_positions.astype(mx.int32), (query_count,))
    flat_key_positions = mx.reshape(key_positions.astype(mx.int32), (T,))

    block_output, block_max, block_sum = _turboquant_fused_attention_kernel(
        inputs=[
            flat_queries,
            flat_key_indices,
            flat_key_norms,
            key_centroids.astype(mx.float32),
            flat_query_positions,
            flat_key_positions,
            flat_value_indices,
            flat_value_norms,
            value_centroids.astype(mx.float32),
        ],
        template=[
            ("OutT", mx.float32),
            ("KD", KD),
            ("VD", VD),
            ("T", T),
            ("R", R),
            ("LQ", LQ),
            ("Causal", int(causal)),
        ],
        grid=(32, query_count, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(query_count, VD), (query_count,), (query_count,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return (
        mx.reshape(block_output, (B, Hkv, R, LQ, VD)),
        mx.reshape(block_max, (B, Hkv, R, LQ, 1)),
        mx.reshape(block_sum, (B, Hkv, R, LQ, 1)),
    )
