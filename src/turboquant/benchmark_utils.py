from __future__ import annotations

import re
import string
from typing import Iterable, Optional, Sequence


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_INTEGER_RE = re.compile(r"-?\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def bytes_to_gb(value: int) -> float:
    return float(value) / 1e9


def cache_nbytes(cache: Sequence[object]) -> int:
    return sum(int(getattr(entry, "nbytes", 0)) for entry in cache)


def normalize_answer(text: object) -> str:
    normalized = str(text).casefold().translate(_PUNCT_TRANSLATION)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def exact_match(prediction: object, answers: Iterable[object]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return False
    return any(normalized_prediction == normalize_answer(answer) for answer in answers)


def contains_answer(prediction: object, answers: Iterable[object]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return False
    return any(
        normalized_answer and normalized_answer in normalized_prediction
        for normalized_answer in (normalize_answer(answer) for answer in answers)
    )


def first_integer(text: object) -> Optional[int]:
    match = _INTEGER_RE.search(str(text))
    if match is None:
        return None
    return int(match.group(0))


def matches_integer(prediction: object, answers: Iterable[object]) -> bool:
    predicted_value = first_integer(prediction)
    if predicted_value is None:
        return False
    reference_values = {first_integer(answer) for answer in answers}
    reference_values.discard(None)
    return predicted_value in reference_values


def encode_chat_prompt(
    tokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        except TypeError:
            pass
    return tokenizer.encode(user_prompt)
