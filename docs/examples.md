# Examples Guide

Zoo-Keeper ships runnable example programs under [`examples/`](../examples/).
That directory is the source of truth for end-to-end usage. This page maps
topics to binaries and keeps only short API illustrations.

**Start here:** [`examples/README.md`](../examples/README.md) — build flags,
usage lines, and a documentation map.

## Build

```bash
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

## Programs at a glance

| Binary | Topic |
|--------|--------|
| [`minimal_agent`](../examples/minimal_agent.cpp) | First agent, split configs |
| [`demo_chat`](../examples/demo_chat.cpp) | Chat loop, tools, streaming, metrics |
| [`demo_extract`](../examples/demo_extract.cpp) | Structured `extract()` |
| [`model_generate`](../examples/model_generate.cpp) | Sync `zoo::core::Model` |
| [`error_handling`](../examples/error_handling.cpp) | `Expected` error paths |
| [`stream_cancel`](../examples/stream_cancel.cpp) | Streaming + cancellation |
| [`manual_tool_schema`](../examples/manual_tool_schema.cpp) | Manual schema tools + `tool_trace` |

## JSON config (`demo_chat`)

`demo_chat` loads a top-level JSON wrapper with nested `model`, `agent`, and
`generation` blocks. See [`examples/config.example.json`](../examples/config.example.json).
The library serializes the same structs through `zoo/core/json.hpp` — see
[Configuration](configuration.md).

## API sketches

The fragments below are not complete programs. Open the linked source file to
run the full flow.

### Streaming

Pass a token callback to `chat()` — see [`stream_cancel.cpp`](../examples/stream_cancel.cpp)
and streaming inside [`demo_chat.cpp`](../examples/demo_chat.cpp).

```cpp
auto handle = agent->chat(
    zoo::MessageView{zoo::Role::User, "Write a haiku"},
    zoo::GenerationOverride::inherit_defaults(),
    [](std::string_view token) { std::cout << token << std::flush; });
auto response = handle.await_result();
```

### Multi-turn history

Retained history is automatic — try two turns in `demo_chat` or
[`minimal_agent.cpp`](../examples/minimal_agent.cpp) extended locally.

### Tools and `tool_trace`

Typed registration and the agentic loop: [`demo_chat.cpp`](../examples/demo_chat.cpp).
Manual schemas and trace printing: [`manual_tool_schema.cpp`](../examples/manual_tool_schema.cpp).
Deep dive: [Tools](tools.md).

### Cancellation

```cpp
auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "Write a long essay"});
handle.cancel();
```

Runnable: [`stream_cancel.cpp`](../examples/stream_cancel.cpp).

### Structured extraction

```cpp
auto handle = agent->extract(schema, "There are 7 apples on the shelf.");
auto response = handle.await_result();
```

Runnable: [`demo_extract.cpp`](../examples/demo_extract.cpp). Schema reference:
[extract.md](extract.md).

### Response metrics

`TextResponse::metrics` and `usage` are printed after each turn in `demo_chat`.

### Sync `Model` (no agent thread)

Runnable: [`model_generate.cpp`](../examples/model_generate.cpp).

## See Also

- [Getting Started](getting-started.md) — setup walkthrough
- [Tools](tools.md) — tool system
- [Configuration](configuration.md) — config fields
- [Building](building.md) — platform and integration notes
