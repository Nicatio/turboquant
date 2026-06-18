from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


def _unix_time() -> int:
    return int(time.time())


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def normalize_chat_messages(messages: Iterable[Dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "tool":
            tool_name = str(
                message.get("name")
                or message.get("tool_call_id")
                or "tool"
            )
            normalized.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n"
                        f"{_extract_text_content(message.get('content'))}"
                    ),
                }
            )
            continue

        content = _extract_text_content(message.get("content"))
        if role == "assistant" and not content and message.get("tool_calls"):
            content = _render_tool_call_history(message["tool_calls"])
        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )
    return normalized


def _format_tool_schema(tools: Iterable[Dict[str, Any]]) -> str:
    lines = [
        "You may call a tool when needed.",
        "If you call a tool, reply with exactly this format and no extra prose:",
        "<|tool_call>call:TOOL_NAME{\"arg\":\"value\"}<tool_call|>",
        "Available tools:",
    ]
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", {})
        name = function.get("name", "tool")
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        lines.append(f"- {name}: {description}")
        lines.append(
            "  parameters: "
            + json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n".join(lines)


def render_chat_prompt(tokenizer, messages: Iterable[Dict[str, Any]], tools: Optional[Iterable[Dict[str, Any]]] = None) -> str:
    normalized = normalize_chat_messages(messages)
    if tools:
        tool_message = {"role": "system", "content": _format_tool_schema(tools)}
        if normalized and normalized[0]["role"] == "system":
            normalized[0] = {
                "role": "system",
                "content": normalized[0]["content"] + "\n\n" + tool_message["content"],
            }
        else:
            normalized.insert(0, tool_message)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
    lines = []
    for message in normalized:
        role = message["role"].upper()
        lines.append(f"{role}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


_THOUGHT_CHANNEL_PATTERN = re.compile(
    r"^\s*<\|channel\>thought\n.*?<channel\|>\s*",
    re.DOTALL,
)


def strip_thought_channel(text: str) -> str:
    return _THOUGHT_CHANNEL_PATTERN.sub("", text, count=1).lstrip()


_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<\|tool_call\>\s*(.*?)\s*<tool_call\|>",
    re.DOTALL,
)


def _render_tool_call_history(tool_calls: Iterable[Dict[str, Any]]) -> str:
    rendered = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = str(function.get("name", "tool"))
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        rendered.append(f"<|tool_call>call:{name}{arguments}<tool_call|>")
    return "\n".join(rendered)


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    tool_calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_BLOCK_PATTERN.finditer(text):
        inner = match.group(1).strip()
        if not inner.startswith("call:"):
            continue
        payload = inner[len("call:") :].strip()
        brace_index = payload.find("{")
        if brace_index == -1:
            continue
        name = payload[:brace_index].strip()
        arguments = payload[brace_index:].strip()
        try:
            json.loads(arguments)
        except json.JSONDecodeError:
            continue
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    remaining = _TOOL_CALL_BLOCK_PATTERN.sub("", text).strip()
    return tool_calls, remaining


def gemma4_args(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "args"):
        return model.language_model.args
    return model.args


def build_gemma4_cache(
    model,
    *,
    implementation: str,
    bits: float,
    seed: int,
    block_size: int,
    recent_window: int,
    recent_slack: int,
    dense_shadow: bool,
    share_quantizers: bool,
):
    from mlx_lm.models.cache import RotatingKVCache

    from turboquant.kv_cache import (
        TurboQuantDirectKVCache,
        TurboQuantKVCache,
        TurboQuantQuantizerPool,
    )

    args = gemma4_args(model)
    first_kv_shared = args.num_hidden_layers - args.num_kv_shared_layers
    layer_types = args.layer_types
    quantizer_pool = TurboQuantQuantizerPool() if share_quantizers else None
    caches = []
    for i in range(first_kv_shared):
        if layer_types[i] == "full_attention":
            if implementation == "direct":
                caches.append(
                    TurboQuantDirectKVCache(
                        bits=bits,
                        seed=seed + i,
                        compute_stats=False,
                        block_size=block_size,
                        recent_window_tokens=recent_window,
                        recent_slack_tokens=recent_slack,
                        quantizer_pool=quantizer_pool,
                    )
                )
            else:
                caches.append(
                    TurboQuantKVCache(
                        bits=bits,
                        seed=seed + i,
                        compute_stats=False,
                        use_dense_shadow=dense_shadow,
                        recent_window_tokens=recent_window,
                        recent_slack_tokens=recent_slack,
                        quantizer_pool=quantizer_pool,
                    )
                )
        else:
            caches.append(
                RotatingKVCache(
                    max_size=args.sliding_window,
                    keep=0,
                )
            )
    return caches


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: Optional[int] = Field(default=256, alias="max_completion_tokens")
    stream: bool = False
    stop: Optional[str | list[str]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Any = None

    model_config = {"populate_by_name": True}


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 0.0
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[str | list[str]] = None


@dataclass
class ServerConfig:
    model_ref: str
    served_model_name: str
    implementation: str
    bits: float
    seed: int
    block_size: int
    recent_window: int
    recent_slack: int
    dense_shadow: bool
    share_quantizers: bool
    prefill_step_size: int
    api_key: Optional[str]


class Gemma4OpenAIBackend:
    def __init__(self, config: ServerConfig):
        from mlx_lm import load

        from turboquant.hf_cache import resolve_cached_model_path
        from turboquant.mlx_attention import enable_turboquant_gemma4_attention

        self.config = config
        resolved_model = resolve_cached_model_path(config.model_ref)
        self.resolved_model = resolved_model
        self.model, self.tokenizer = load(resolved_model)
        if config.implementation == "direct":
            enable_turboquant_gemma4_attention(self.model)
        self._lock = threading.Lock()

    def _build_cache(self):
        if self.config.implementation == "baseline":
            return self.model.make_cache()
        return build_gemma4_cache(
            self.model,
            implementation=self.config.implementation,
            bits=self.config.bits,
            seed=self.config.seed,
            block_size=self.config.block_size,
            recent_window=self.config.recent_window,
            recent_slack=self.config.recent_slack,
            dense_shadow=self.config.dense_shadow,
            share_quantizers=self.config.share_quantizers,
        )

    def _truncate_stop(self, text: str, stop: Optional[str | list[str]]) -> tuple[str, Optional[str]]:
        if not stop:
            return text, None
        stops = [stop] if isinstance(stop, str) else list(stop)
        earliest = None
        matched = None
        for token in stops:
            idx = text.find(token)
            if idx == -1:
                continue
            if earliest is None or idx < earliest:
                earliest = idx
                matched = token
        if earliest is None:
            return text, None
        return text[:earliest], matched

    def generate_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: Optional[str | list[str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = render_chat_prompt(self.tokenizer, messages, tools=tools)
        sampler = make_sampler(temp=temperature)
        cache = self._build_cache()
        with self._lock:
            text = ""
            final_response = None
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=self.config.prefill_step_size,
            ):
                text += response.text
                final_response = response
            if final_response is None:
                raise RuntimeError("Generation did not produce a response.")

        text, matched_stop = self._truncate_stop(text, stop)
        text = strip_thought_channel(text)
        tool_calls, text = parse_tool_calls(text)
        finish_reason = "tool_calls" if tool_calls else ("stop" if matched_stop is not None else final_response.finish_reason)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "prompt_tokens": int(final_response.prompt_tokens),
            "completion_tokens": int(final_response.generation_tokens),
            "finish_reason": finish_reason or "stop",
        }

    def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: Optional[str | list[str]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Iterator[str]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = render_chat_prompt(self.tokenizer, messages, tools=tools)
        sampler = make_sampler(temp=temperature)
        cache = self._build_cache()
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = _unix_time()

        initial = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.served_model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(initial)}\n\n"
        with self._lock:
            text = ""
            final_response = None
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=self.config.prefill_step_size,
            ):
                text += response.text
                final_response = response
            if final_response is None:
                raise RuntimeError("Generation did not produce a response.")

        text, matched_stop = self._truncate_stop(text, stop)
        text = strip_thought_channel(text)
        tool_calls, text = parse_tool_calls(text)

        if tool_calls:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            finish_reason = "tool_calls"
        else:
            if text:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.config.served_model_name,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            finish_reason = "stop" if matched_stop is not None else (final_response.finish_reason or "stop")

        done_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.served_model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _require_api_key(expected_key: Optional[str], authorization: Optional[str]) -> None:
    if not expected_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    supplied = authorization.removeprefix("Bearer ").strip()
    if supplied != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _chat_completion_response(
    model_name: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    created = _unix_time()
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _completion_response(model_name: str, text: str, prompt_tokens: int, completion_tokens: int, finish_reason: str) -> dict[str, Any]:
    created = _unix_time()
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_openai_compatible_app(config: ServerConfig) -> FastAPI:
    backend = Gemma4OpenAIBackend(config)
    app = FastAPI(title="TurboQuant Gemma4 OpenAI-Compatible Server")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": config.served_model_name,
            "resolved_model": backend.resolved_model,
            "implementation": config.implementation,
            "bits": config.bits,
        }

    @app.get("/v1/models")
    def list_models(authorization: Optional[str] = Header(default=None)):
        _require_api_key(config.api_key, authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": config.served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(default=None),
    ):
        _require_api_key(config.api_key, authorization)
        if request.model != config.served_model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'.")

        max_tokens = request.max_tokens if request.max_tokens is not None else 256
        messages = [message.model_dump() for message in request.messages]

        if request.stream:
            return StreamingResponse(
                backend.stream_chat(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=max_tokens,
                    stop=request.stop,
                    tools=request.tools,
                ),
                media_type="text/event-stream",
            )

        result = backend.generate_chat(
            messages=messages,
            temperature=request.temperature,
            max_tokens=max_tokens,
            stop=request.stop,
            tools=request.tools,
        )
        return JSONResponse(
            _chat_completion_response(
                config.served_model_name,
                result["text"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["finish_reason"],
                result.get("tool_calls"),
            )
        )

    @app.post("/v1/completions")
    def completions(
        request: CompletionRequest,
        authorization: Optional[str] = Header(default=None),
    ):
        _require_api_key(config.api_key, authorization)
        if request.model != config.served_model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model '{request.model}'.")

        prompt_text = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)
        result = backend.generate_chat(
            messages=[{"role": "user", "content": prompt_text}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
        return JSONResponse(
            _completion_response(
                config.served_model_name,
                result["text"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["finish_reason"],
            )
        )

    return app
