# Plan: `demo_chat` Generic Tool-Calling Follow-Up

## Summary

The first native grammar crash is fixed, but `demo_chat` still has two follow-up issues when Gemma tools are enabled:

1. Normal assistant replies are surfaced as wrapper JSON like `{"response": ...}`.
2. A successful tool call crashes on the next turn with Gemma's Jinja alternation error.

This is not another grammar sampler bug. The current problem is that Gemma lands in llama.cpp's `COMMON_CHAT_FORMAT_GENERIC` tool-calling fallback, and Zoo-Keeper is treating that fallback as if it were a fully symmetric native tool format.

The recommended fix is Zoo-Keeper-local:

- keep the existing structured native path unchanged for non-generic formats
- special-case `COMMON_CHAT_FORMAT_GENERIC` in the runtime
- unwrap generic `response` payloads before showing them to users
- stop round-tripping literal `Role::Tool` messages through strict user/assistant-only templates like Gemma

Do not patch vendored `llama.cpp` first unless the Zoo-Keeper-side fix still reproduces the issue.

## Reproduction

### Observed normal-response issue

```text
You: Please write me hello world in C++

Assistant: {
  "response": "```cpp\n#include <iostream>\n\nint main() {\n  std::cout << \"Hello, world!\" << std::endl;\n  return 0;\n}\n```"
}
```

### Observed tool-loop crash

```text
You: What is the sum of 88 and 75 multiplied by 11? Use tools

Assistant: {
  "tool_call": {
    "name": "...",
    "arguments": { ... }
  }
}
Error: [302] common_chat_templates_apply failed:
Jinja Exception: Conversation roles must alternate user/assistant/user/assistant/...
```

### Local repro used during investigation

```bash
printf 'Please write me hello world in C++\nWhat is the sum of 88 and 75 multiplied by 11? Use tools\n/quit\n' | ./build/examples/demo_chat examples/config.example.json
```

## Confirmed Findings

### 1. Gemma-with-tools is using llama.cpp's generic tool fallback

Vendored llama.cpp explicitly tests that Gemma with tools resolves to `COMMON_CHAT_FORMAT_GENERIC` rather than a model-specific native handler:

- `extern/llama.cpp/tests/test-chat.cpp`
- `src/core/model_tool_calling.cpp`

This means the JSON wrapper is expected unless Zoo-Keeper unwraps it.

### 2. Generic format intentionally emits JSON wrappers

`common_chat_params_init_generic(...)` in `extern/llama.cpp/common/chat.cpp` builds a schema that requires either:

- `{"tool_call": ...}` or `{"tool_calls": ...}`
- `{"response": ...}`

It also injects an instruction telling the model to respond in JSON format. So the first issue is not a model formatting accident.

### 3. Zoo-Keeper only unwraps native output when a tool call was detected

In `src/agent/runtime_tool_loop.cpp`:

- native tool output is parsed only when `generated->tool_call_detected` is true
- otherwise the raw generated text is used as `response_text`

That is why generic `{"response": ...}` messages leak through unchanged.

### 4. Native streaming currently forwards raw generic JSON to the user

Also in `src/agent/runtime_tool_loop.cpp`, the native path streams token fragments directly to the caller callback. For `COMMON_CHAT_FORMAT_GENERIC`, those token fragments are the JSON wrapper itself.

Even if final response parsing is fixed, `demo_chat` will still print raw JSON unless streaming is handled separately for the generic path.

### 5. The second crash is a history-shape problem, not another grammar failure

After a tool call, Zoo-Keeper currently appends:

- an assistant message with structured tool-call metadata
- a `Role::Tool` message with the tool result

The next generation calls `common_chat_templates_apply(...)` via `src/core/model_prompt.cpp`.

Gemma's template rejects:

- system role directly
- any non-alternating user/assistant sequence
- any literal `tool` role in the conversation

So the next render fails before generation starts.

### 6. Generic fallback is not symmetric for history

Before rendering, llama.cpp's generic fallback rewrites assistant `tool_calls` into opaque JSON appended to assistant content. That makes the assistant tool-call message usable.

But it does not rewrite `role="tool"` messages into a form that strict templates like Gemma can safely render. Upstream tests also note that generic tool-calling is not symmetric.

### 7. This is primarily a Zoo-Keeper integration problem

The earlier grammar-stack crash came from sampler ordering and stale grammar refresh. That part is fixed.

The current issues come from how Zoo-Keeper consumes and replays llama.cpp generic fallback output:

- it exposes generic wrapper JSON directly to the user
- it stores tool-result history in a shape Gemma's template cannot replay

## Recommended Implementation

### Decision

Keep `COMMON_CHAT_FORMAT_GENERIC` enabled as a constrained generation mode, but stop treating it like a fully symmetric native multi-turn tool format.

### Internal interface changes

No public API changes are needed.

Add one small internal distinction between:

- structured native tool-calling formats
- generic fallback tool-calling

Recommended shape:

- add an internal enum or flag on the model/backend boundary
- avoid relying on `tool_calling_format_name()` string comparisons in the runtime

The runtime needs this so it can branch cleanly without coupling itself to llama.cpp format-name strings.

### Runtime behavior for structured native formats

Leave the current post-fix native path unchanged for non-generic formats.

Those formats are the ones where the template and parser already understand tool-call and tool-result history in a model-specific way.

### Runtime behavior for generic fallback

#### Output parsing

- always parse final native output for the generic path
- if the parsed result contains `response`, surface only the parsed content
- if the parsed result contains `tool_call`, continue into the tool loop

Do not rely on `tool_call_detected` as the only gate for whether native output should be parsed.

#### Streaming

- do not forward raw generic tokens directly to the user callback
- buffer generation for generic native turns
- once generation completes, emit only parsed user-visible content to the callback
- if the turn is a tool call, emit nothing user-visible from the tool-call JSON itself

This preserves a clean `demo_chat` experience even when the model is constrained by the generic schema.

#### History shaping after a generic tool call

Keep the assistant tool-call message, because generic fallback can rewrite assistant `tool_calls` into assistant content safely.

Do not append a literal `Role::Tool` message for the generic path.

Instead, after tool execution or validation failure, append a generated `Role::User` follow-up that contains:

- the tool result or validation error
- enough context to tell the model to continue and answer the original request

This preserves Gemma's required user/assistant alternation:

```text
user -> assistant(tool_call) -> user(tool result follow-up) -> assistant(final answer)
```

That same pattern also supports multiple tool iterations:

```text
user -> assistant(tool_call) -> user(tool result) -> assistant(tool_call) -> user(tool result) -> assistant(final)
```

### Suggested generic follow-up message shape

Keep the message simple and deterministic. For example:

```text
Tool result for `multiply`:
{"result":6600}

Use this tool result to continue answering the original request.
Call another tool only if still necessary.
```

For validation failures:

```text
Tool call validation failed for `multiply`:
Error: ...

Please correct the arguments and continue.
```

The exact wording should stay stable so tests can assert against it.

## Files Most Likely to Change Tomorrow

- `src/agent/runtime_tool_loop.cpp`
- `src/agent/backend_model.cpp`
- `include/zoo/internal/agent/backend.hpp`

Possibly:

- `src/core/model_tool_calling.cpp`
- `include/zoo/core/model.hpp`

Only if the internal flavor enum or flag is exposed from the core model side.

## Test Plan

### Unit tests

Add focused runtime tests that cover generic behavior without depending on a live model:

1. Generic native `{"response": ...}` output is unwrapped before it becomes the final response text.
2. Generic native streaming does not forward raw wrapper JSON to the user callback.
3. Generic tool execution does not append `Role::Tool`; it appends a user follow-up instead.
4. Generic validation failure also uses the user follow-up path and preserves alternation.

### Integration tests

Extend the live-model integration coverage with a Gemma-style sequence:

1. tools enabled
2. fenced-code prompt returns normal plain assistant text, not wrapper JSON
3. tool prompt completes successfully without the Jinja alternation crash
4. final answer is surfaced as assistant text rather than raw generic JSON

### Manual verification

Run:

```bash
./build/examples/demo_chat examples/config.example.json
```

Check:

1. `Please write me hello world in C++`
2. `What is the sum of 88 and 75 multiplied by 11? Use tools`

Expected:

- first reply appears as normal assistant text
- no raw `{"response": ...}` output is printed
- a tool turn does not expose raw `{"tool_call": ...}` output to the user
- no `Error: [302] common_chat_templates_apply failed`

## Assumptions

- No public API changes under `include/zoo/*.hpp`
- Vendored `llama.cpp` stays unchanged for this fix
- Existing unrelated worktree changes remain untouched:
  - `examples/config.example.json`
  - `.agents/`
  - `ai-first-project-design-guide.md`

## Recommended Commit

For the doc-only handoff:

```text
document generic tool-calling follow-up
```
