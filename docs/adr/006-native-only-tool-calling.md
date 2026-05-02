# ADR-006: Native-Only Tool Calling (No Generic Fallback)

## Status

Accepted

## Context

Zoo-Keeper's tool calling system delegates format detection to llama.cpp's `common_chat_templates_apply()`. As of llama.cpp b8992, native tool calling is represented by PEG parser variants such as `COMMON_CHAT_FORMAT_PEG_SIMPLE`, `COMMON_CHAT_FORMAT_PEG_NATIVE`, and `COMMON_CHAT_FORMAT_PEG_GEMMA4`.

Earlier llama.cpp versions also exposed a generic JSON wrapper fallback for models whose templates did not define a native tool format. Supporting that fallback required a parallel runtime code path: streaming suppression of wrapper JSON, `Role::User` messages instead of `Role::Tool` for history, and special parsing/unwrapping logic.

## Decision

`Model::set_tool_calling()` rejects models that resolve to `COMMON_CHAT_FORMAT_CONTENT_ONLY`. For these models, `set_tool_calling()` returns `false` and tool calling is disabled.

The runtime has a single tool calling code path: llama.cpp PEG parser formats only. Tool results are always injected as `Role::Tool` messages with `tool_call_id`. The stored parser state includes llama.cpp's `generation_prompt`, because b8992 prepends that prompt before parsing generated assistant text.

## Rationale

- **Simplicity:** A single code path is easier to reason about, test, and maintain. Eliminating the generic fallback removed ~750 lines of dual-path branching.
- **Reliability:** The removed generic fallback wrapped all output in JSON, making it hard to distinguish tool calls from normal text without heuristics. PEG formats have explicit grammar triggers and parsers.
- **Streaming fidelity:** Native formats stream tokens directly to callers. The generic format required buffering and suppressing wrapper JSON, adding latency and complexity.
- **History correctness:** Native formats support `Role::Tool` messages for proper round-tripping through chat templates. The generic format required injecting tool results as `Role::User` messages, which pollutes the conversation history.
- **Upstream owns format growth:** llama.cpp's PEG parser variants cover the model-specific formatting details. When upstream adds another native parser variant, Zoo-Keeper can usually consume it without runtime branching.

## Consequences

- Models without a native tool calling format (e.g., some Gemma variants) cannot use tool calling through Zoo-Keeper. They work fine for normal chat.
- If llama.cpp adds native format support for a model family in the future, Zoo-Keeper picks it up automatically (no code changes needed).
- `ToolCallParser` remains available for consumers who want heuristic tool call detection outside the agent runtime.
- Adding a new tool calling path in the future would require explicit opt-in, not silent fallback.
