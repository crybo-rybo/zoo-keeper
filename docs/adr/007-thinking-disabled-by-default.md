# ADR-007: Thinking Disabled by Default

## Status

Accepted (temporary — must be revisited)

## Context

Some models (e.g. DeepSeek-R1, QwQ, certain Qwen variants) support a "thinking" mode where
the model emits an internal reasoning trace (delimited by `<think>` / `</think>` or
`<|START_THINKING|>` / `<|END_THINKING|>`) before its final response.

llama.cpp's `common_chat_templates_inputs` exposes two fields for this:

- `enable_thinking` (bool, default `true`) — when `false`, the rendered prompt suppresses
  the thinking block (e.g. by injecting `<|START_THINKING|><|END_THINKING|>` immediately).
- `reasoning_format` (enum) — controls how thinking content is extracted from raw output
  during parsing (`NONE`, `AUTO`, `DEEPSEEK`, `DEEPSEEK_LEGACY`).

`common_chat_msg` also carries a `reasoning_content` field alongside `content` for the
parsed-out thinking trace.

Zoo-Keeper currently has no public API surface for thinking: `Response` has no
`reasoning_content` field, there is no streaming path for thinking tokens (they would
appear inline in the token stream before content), and the `Config` struct exposes no
thinking-related options.

## Decision

Set `inputs.enable_thinking = false` in every `common_chat_templates_apply()` call inside
`model_prompt.cpp` until a proper thinking API is designed and implemented.

This is applied in both `render_prompt_delta()` (generation) and `finalize_response()`
(KV-cache bookkeeping) to keep the two render passes consistent.

## Consequences

- Thinking models behave like standard instruction-tuned models: no reasoning trace is
  emitted, output is purely the final answer.
- No behaviour change for non-thinking models (the flag is a no-op when the template does
  not support it).
- Integration and development testing of thinking models is unaffected by reasoning noise.

## What Must Be Done Before Removing This

1. Add `reasoning_content` to `zoo::Response` (Layer 3) and wire it through from
   `common_chat_msg::reasoning_content` in the parse path.
2. Decide on a streaming strategy — either a second `on_thinking_token` callback, or a
   filter similar to `ToolCallWordTriggerFilter` that separates thinking from content tokens.
3. Expose `enable_thinking` and `reasoning_format` in `zoo::Config` (or a nested
   `ThinkingConfig` struct).
4. Call `common_chat_templates_support_enable_thinking()` after model load and surface the
   result so callers know whether the loaded model supports the flag at all.
5. Remove the hard-coded `inputs.enable_thinking = false` lines from `model_prompt.cpp`.
