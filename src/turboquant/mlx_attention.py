from __future__ import annotations

from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn

from turboquant.kv_cache import TurboQuantDirectKVCache


def _supports_direct_attention(cache: Any) -> bool:
    return isinstance(cache, TurboQuantDirectKVCache)


def get_transformer_layers(model: Any) -> Iterable[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise TypeError("Unsupported model structure: could not find transformer layers.")


class TurboQuantLlamaAttention(nn.Module):
    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.n_heads = base_attention.n_heads
        self.n_kv_heads = base_attention.n_kv_heads
        self.head_dim = base_attention.head_dim
        self.scale = base_attention.scale
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.rope = base_attention.rope

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None and _supports_direct_attention(cache):
            query_start = cache.offset
            queries = self.rope(queries, offset=query_start)
            keys = self.rope(keys, offset=query_start)
            cache.append(keys, values)
            output = cache.direct_attention(
                queries,
                scale=self.scale,
                mask=mask,
                query_start=query_start,
            )
        else:
            from mlx_lm.models.base import scaled_dot_product_attention

            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TurboQuantQwen3_5Attention(nn.Module):
    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.num_key_value_heads = base_attention.num_key_value_heads
        self.num_attention_heads = base_attention.num_attention_heads
        self.head_dim = base_attention.head_dim
        self.scale = base_attention.scale
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.rotary_emb = base_attention.rotary_emb

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        from mlx_vlm.models.base import scaled_dot_product_attention
        from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb

        B, L, _ = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]
        if position_ids is None:
            cache_offset = cache.offset if cache is not None else 0
            kv_seq_len += cache_offset + 1
            position_ids = mx.arange(cache_offset, cache_offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        elif cache is not None:
            kv_seq_len += cache.offset + 1

        cos, sin = self.rotary_emb(values, position_ids)

        if mask is not None and isinstance(mask, mx.array):
            if isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None and _supports_direct_attention(cache):
            query_start = cache.offset
            cache.append(keys, values)
            output = cache.direct_attention(
                queries,
                scale=self.scale,
                mask=mask,
                query_start=query_start,
            )
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class TurboQuantQwen3VLAttention(nn.Module):
    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.n_heads = base_attention.n_heads
        self.n_kv_heads = base_attention.n_kv_heads
        self.head_dim = base_attention.head_dim
        self.scale = base_attention.scale
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.rotary_emb = base_attention.rotary_emb

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        from mlx_vlm.models.base import scaled_dot_product_attention
        from mlx_vlm.models.qwen3_vl.language import apply_multimodal_rotary_pos_emb

        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        kv_seq_len = keys.shape[-2]
        cache_offset = cache.offset if cache is not None else 0
        if position_ids is None:
            kv_seq_len += cache_offset + 1
            position_ids = mx.arange(cache_offset, cache_offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
        elif cache is not None:
            kv_seq_len += cache_offset + 1

        cos, sin = self.rotary_emb(values, position_ids)

        if mask is not None and isinstance(mask, mx.array):
            if isinstance(kv_seq_len, mx.array):
                kv_seq_len = kv_seq_len.max().item()
            mask = mask[..., : int(kv_seq_len)]

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None and _supports_direct_attention(cache):
            query_start = cache.offset
            cache.append(keys, values)
            output = cache.direct_attention(
                queries,
                scale=self.scale,
                mask=mask,
                query_start=query_start,
            )
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TurboQuantGemma4Attention(nn.Module):
    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.head_dim = base_attention.head_dim
        self.n_heads = base_attention.n_heads
        self.n_kv_heads = base_attention.n_kv_heads
        self.scale = base_attention.scale
        self.use_k_eq_v = base_attention.use_k_eq_v
        self.is_sliding = getattr(base_attention, "is_sliding", False)
        config = getattr(base_attention, "config", None)
        self.sliding_window = getattr(config, "sliding_window", None)
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = getattr(base_attention, "v_proj", None)
        self.o_proj = base_attention.o_proj
        self.q_norm = base_attention.q_norm
        self.k_norm = base_attention.k_norm
        self.v_norm = base_attention.v_norm
        self.rope = base_attention.rope

    def _align_mask_for_keys(
        self,
        mask: mx.array | None,
        *,
        query_length: int,
        key_length: int,
    ) -> mx.array | None:
        if mask is None or not isinstance(mask, mx.array):
            return mask
        if mask.shape[-1] == key_length:
            return mask

        from mlx_lm.models.base import create_causal_mask

        offset = max(key_length - query_length, 0)
        window_size = self.sliding_window if self.is_sliding else None
        return create_causal_mask(query_length, offset=offset, window_size=window_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        shared_kv: tuple[mx.array, mx.array] | None = None,
        offset: Any | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], Any]:
        from mlx_lm.models.base import scaled_dot_product_attention

        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
            rope_offset = offset if offset is not None else 0
            query_start = int(offset) if offset is not None else 0
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            values = keys
            if not self.use_k_eq_v:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            query_start = cache.offset if cache is not None else 0
            rope_offset = mx.array(query_start) if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=rope_offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=rope_offset)

        if shared_kv is None and cache is not None and _supports_direct_attention(cache):
            cache.append(keys, values)
            output = cache.direct_attention(
                queries,
                scale=self.scale,
                mask=mask,
                query_start=query_start,
            )
            dense_keys, dense_values = cache.decoded_state
            shared_kv = (dense_keys, dense_values)
        else:
            if shared_kv is None and cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
                shared_kv = (keys, values)

            mask = self._align_mask_for_keys(
                mask,
                query_length=queries.shape[-2],
                key_length=keys.shape[-2],
            )

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            if shared_kv is None:
                shared_kv = (keys, values)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), shared_kv, rope_offset


def _get_full_attention_layers(model: Any) -> list[Any]:
    return [layer for layer in get_transformer_layers(model) if hasattr(layer, "self_attn")]


def enable_turboquant_direct_attention(model: Any) -> Any:
    for layer in _get_full_attention_layers(model):
        if isinstance(layer.self_attn, TurboQuantLlamaAttention):
            continue
        layer.self_attn = TurboQuantLlamaAttention(layer.self_attn)
    return model


def enable_turboquant_qwen3_5_attention(model: Any) -> Any:
    for layer in _get_full_attention_layers(model):
        if getattr(layer, "is_linear", False):
            continue
        if isinstance(layer.self_attn, TurboQuantQwen3_5Attention):
            continue
        layer.self_attn = TurboQuantQwen3_5Attention(layer.self_attn)
    return model


def enable_turboquant_qwen3_vl_attention(model: Any) -> Any:
    for layer in _get_full_attention_layers(model):
        if isinstance(layer.self_attn, TurboQuantQwen3VLAttention):
            continue
        layer.self_attn = TurboQuantQwen3VLAttention(layer.self_attn)
    return model


def enable_turboquant_qwen_vlm_attention(model: Any) -> Any:
    layers = _get_full_attention_layers(model)
    if not layers:
        return model

    module_name = type(layers[0].self_attn).__module__
    if "qwen3_vl" in module_name:
        return enable_turboquant_qwen3_vl_attention(model)
    return enable_turboquant_qwen3_5_attention(model)


def enable_turboquant_gemma4_attention(model: Any) -> Any:
    for layer in _get_full_attention_layers(model):
        if isinstance(layer.self_attn, TurboQuantGemma4Attention):
            continue
        layer.self_attn = TurboQuantGemma4Attention(layer.self_attn)
    return model
