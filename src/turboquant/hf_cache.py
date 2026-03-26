from __future__ import annotations

from pathlib import Path


def resolve_cached_model_path(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_ref

    try:
        cached = snapshot_download(model_ref, local_files_only=True)
    except Exception:
        return model_ref

    cached_path = Path(cached)
    if cached_path.exists():
        return str(cached_path)
    return model_ref
