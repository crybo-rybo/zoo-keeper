# ADR-006: Native-Only Tool Calling (No Generic Fallback)

## Status

Accepted

## Context

Zoo-Keeper's tool calling system delegates format detection to llama.cpp's `common_chat_templates_apply()`, which supports 29+ model-specific tool calling formats. llama.cpp also provides a generic JSON wrapper format (`COMMON_CHAT_FORMAT_GENERIC`) as a fallback for models whose chat templates do not define a native tool calling format. This wraps ALL model output in JSON envelopes like `{"response": "..."}` or `{"tool_call": {...}}`.

Supporting the generic format required a parallel runtime code path: streaming suppression of wrapper JSON, `Role::User` messages instead of `Role::Tool` for history (because generic templates don't support the tool role), and special parsing/unwrapping logic. An earlier version of the branch implemented both paths, adding ~200 lines of runtime branching.

## Decision

`Model::set_tool_calling()` rejects models that resolve to `COMMON_CHAT_FORMAT_GENERIC` or `COMMON_CHAT_FORMAT_CONTENT_ONLY`. For these models, `set_tool_calling()` returns `false` and tool calling is disabled.

The runtime has a single tool calling code path: native structured formats only. Tool results are always injected as `Role::Tool` messages with `tool_call_id`. The Layer 2 utilities (`ToolCallParser`, `ToolCallInterceptor`) remain available as standalone utilities but are not used by the agent runtime.

## Rationale

- **Simplicity:** A single code path is easier to reason about, test, and maintain. Eliminating the generic fallback removed ~750 lines of dual-path branching.
- **Reliability:** The generic format wraps ALL output in JSON, making it impossible to reliably distinguish tool calls from normal text responses without heuristics. Native formats have explicit grammar triggers and format-specific parsers.
- **Streaming fidelity:** Native formats stream tokens directly to callers. The generic format required buffering and suppressing wrapper JSON, adding latency and complexity.
- **History correctness:** Native formats support `Role::Tool` messages for proper round-tripping through chat templates. The generic format required injecting tool results as `Role::User` messages, which pollutes the conversation history.
- **Coverage is already high:** The 29+ native formats in llama.cpp cover the vast majority of models that support tool calling (Llama 3.x, Mistral, Hermes, Command-R, Qwen, DeepSeek, etc.). Models that only get the generic format typically have weak tool calling support anyway.

## Consequences

- Models without a native tool calling format (e.g., some Gemma variants) cannot use tool calling through Zoo-Keeper. They work fine for normal chat.
- If llama.cpp adds native format support for a model family in the future, Zoo-Keeper picks it up automatically (no code changes needed).
- The `ToolCallInterceptor` and `ToolCallParser` in `zoo::tools` remain available for consumers who want heuristic tool call detection outside the agent runtime.
- Adding a new tool calling path in the future would require explicit opt-in, not silent fallback.
