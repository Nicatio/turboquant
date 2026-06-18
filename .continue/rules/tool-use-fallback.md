---
name: Tool Use Fallback
description: Prefer real tool usage when available, and never fabricate raw tool-call markup.
alwaysApply: true
---

- Use MCP tools when they are actually available in Agent mode and when a tool would materially help.
- If a tool is unavailable, disabled, or fails, say that plainly in normal prose.
- Never emit raw tool-call markup such as `<|tool_call>...` as part of the answer.
- When the user asks for a summary or explanation, answer directly in plain text unless they explicitly ask you to inspect files or use tools first.
