from __future__ import annotations

from importlib import import_module

__all__ = [
    "EncodedQJL",
    "EncodedVectorMSE",
    "EncodedVectorProd",
    "QJL",
    "RotationOperator",
    "ScalarCodebook",
    "TurboQuantDirectKVCache",
    "TurboQuantKVCache",
    "TurboQuantQuantizerPool",
    "TurboQuantMSE",
    "TurboQuantProd",
    "build_sphere_codebook",
    "enable_turboquant_direct_attention",
    "enable_turboquant_gemma4_attention",
    "enable_turboquant_qwen3_5_attention",
    "enable_turboquant_qwen3_vl_attention",
    "enable_turboquant_qwen_vlm_attention",
    "generate_random_rotation",
    "mean_squared_error",
    "normalize_rows",
    "sample_unit_sphere",
    "squared_inner_product_error",
]


_EXPORTS = {
    "normalize_rows": ("turboquant.datasets", "normalize_rows"),
    "sample_unit_sphere": ("turboquant.datasets", "sample_unit_sphere"),
    "ScalarCodebook": ("turboquant.lloyd_max", "ScalarCodebook"),
    "build_sphere_codebook": ("turboquant.lloyd_max", "build_sphere_codebook"),
    "mean_squared_error": ("turboquant.metrics", "mean_squared_error"),
    "squared_inner_product_error": ("turboquant.metrics", "squared_inner_product_error"),
    "EncodedVectorMSE": ("turboquant.mse_quantizer", "EncodedVectorMSE"),
    "TurboQuantMSE": ("turboquant.mse_quantizer", "TurboQuantMSE"),
    "enable_turboquant_direct_attention": (
        "turboquant.mlx_attention",
        "enable_turboquant_direct_attention",
    ),
    "enable_turboquant_gemma4_attention": (
        "turboquant.mlx_attention",
        "enable_turboquant_gemma4_attention",
    ),
    "enable_turboquant_qwen3_5_attention": (
        "turboquant.mlx_attention",
        "enable_turboquant_qwen3_5_attention",
    ),
    "enable_turboquant_qwen3_vl_attention": (
        "turboquant.mlx_attention",
        "enable_turboquant_qwen3_vl_attention",
    ),
    "enable_turboquant_qwen_vlm_attention": (
        "turboquant.mlx_attention",
        "enable_turboquant_qwen_vlm_attention",
    ),
    "EncodedVectorProd": ("turboquant.prod_quantizer", "EncodedVectorProd"),
    "TurboQuantProd": ("turboquant.prod_quantizer", "TurboQuantProd"),
    "EncodedQJL": ("turboquant.qjl", "EncodedQJL"),
    "QJL": ("turboquant.qjl", "QJL"),
    "TurboQuantDirectKVCache": ("turboquant.kv_cache", "TurboQuantDirectKVCache"),
    "TurboQuantKVCache": ("turboquant.kv_cache", "TurboQuantKVCache"),
    "TurboQuantQuantizerPool": ("turboquant.kv_cache", "TurboQuantQuantizerPool"),
    "RotationOperator": ("turboquant.rotation", "RotationOperator"),
    "generate_random_rotation": ("turboquant.rotation", "generate_random_rotation"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'turboquant' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
