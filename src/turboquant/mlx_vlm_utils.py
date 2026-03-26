from __future__ import annotations

from typing import Any, Tuple

from transformers import AutoProcessor

from mlx_vlm.tokenizer_utils import load_tokenizer
from mlx_vlm.utils import StoppingCriteria, get_model_path, load_config, load_model

from turboquant.hf_cache import resolve_cached_model_path


def load_with_slow_processor(
    model_ref: str,
    trust_remote_code: bool = False,
) -> Tuple[str, Any, Any]:
    resolved_model = resolve_cached_model_path(model_ref)
    model_path = get_model_path(resolved_model)
    model = load_model(model_path, trust_remote_code=trust_remote_code)
    config = load_config(model_path, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )

    detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
    tokenizer_obj = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    processor.detokenizer = detokenizer_class(tokenizer_obj)

    eos_token_id = config.get("eos_token_id", None)
    criteria = StoppingCriteria(
        eos_token_id if eos_token_id is not None else tokenizer_obj.eos_token_ids,
        tokenizer_obj,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.stopping_criteria = criteria
    else:
        processor.stopping_criteria = criteria

    return str(model_path), model, processor
