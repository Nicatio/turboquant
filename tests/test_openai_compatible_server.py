from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant.openai_compatible_server import (
    _chat_completion_response,
    _completion_response,
    _extract_text_content,
    parse_tool_calls,
    normalize_chat_messages,
    render_chat_prompt,
    strip_thought_channel,
)


class _FakeTokenizer:
    chat_template = "fake"

    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        self.last_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        self.last_kwargs.update(kwargs)
        rendered = [f"{m['role']}:{m['content']}" for m in messages]
        if add_generation_prompt:
            rendered.append("assistant:")
        return "\n".join(rendered)


class OpenAICompatibleServerTests(unittest.TestCase):
    def test_extract_text_content_handles_openai_part_list(self) -> None:
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "ignored"}},
            {"type": "text", "text": " world"},
        ]
        self.assertEqual(_extract_text_content(content), "hello world")

    def test_normalize_chat_messages_preserves_roles(self) -> None:
        messages = [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        self.assertEqual(
            normalize_chat_messages(messages),
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_normalize_chat_messages_converts_tool_results(self) -> None:
        messages = [
            {"role": "tool", "name": "read_file", "content": "hello"},
        ]
        self.assertEqual(
            normalize_chat_messages(messages),
            [{"role": "user", "content": "Tool result for read_file:\nhello"}],
        )

    def test_render_chat_prompt_uses_chat_template(self) -> None:
        tokenizer = _FakeTokenizer()
        prompt = render_chat_prompt(
            tokenizer,
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "say hi"},
            ],
        )
        self.assertIn("system:be concise", prompt)
        self.assertIn("user:say hi", prompt)
        self.assertTrue(prompt.endswith("assistant:"))
        self.assertEqual(
            tokenizer.last_kwargs,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )

    def test_render_chat_prompt_includes_tool_schema(self) -> None:
        tokenizer = _FakeTokenizer()
        prompt = render_chat_prompt(
            tokenizer,
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read one file.",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        )
        self.assertIn("Available tools:", prompt)
        self.assertIn("read_file", prompt)
        self.assertIn("<|tool_call>call:TOOL_NAME", prompt)

    def test_strip_thought_channel_removes_internal_reasoning(self) -> None:
        text = "<|channel>thought\nhidden<channel|>\nfinal answer"
        self.assertEqual(strip_thought_channel(text), "final answer")

    def test_parse_tool_calls_extracts_openai_tool_payload(self) -> None:
        text = '<|tool_call>call:read_file{"path":"."}<tool_call|>'
        tool_calls, remaining = parse_tool_calls(text)
        self.assertEqual(remaining, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertEqual(tool_calls[0]["function"]["name"], "read_file")
        self.assertEqual(tool_calls[0]["function"]["arguments"], '{"path":"."}')

    def test_chat_completion_response_shape(self) -> None:
        payload = _chat_completion_response("gemma4-mlx", "hello", 10, 3, "stop")
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(payload["usage"]["total_tokens"], 13)

    def test_chat_completion_response_handles_tool_calls(self) -> None:
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"."}'},
            }
        ]
        payload = _chat_completion_response(
            "gemma4-mlx",
            "",
            10,
            3,
            "tool_calls",
            tool_calls,
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(payload["choices"][0]["message"]["content"])
        self.assertEqual(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "read_file")

    def test_completion_response_shape(self) -> None:
        payload = _completion_response("gemma4-mlx", "hello", 10, 3, "length")
        self.assertEqual(payload["object"], "text_completion")
        self.assertEqual(payload["choices"][0]["text"], "hello")
        self.assertEqual(payload["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
