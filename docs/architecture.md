# Architecture

Zoo-Keeper follows a three-layer architecture with strict dependency direction. Each layer depends only on the layers below it.

## Layer Structure

```
Consumer Code
     |
     v
+-- Layer 3: Agent ---------+
|   zoo::Agent               |   Async orchestration, request queue, agentic tool loop
|   Internal runtime units   |   Mailbox, request tracking, private model backend seam
|   RequestHandle            |   Future-based response handle
+----------------------------+
     |
     v
+-- Layer 2: Tools ----------+
|   zoo::tools::ToolRegistry  |   Tool definitions, schema generation, invocation
|   zoo::tools::ToolCallParser |   Tool call detection in model output
|   zoo::tools::ToolArgumentsValidator |   Argument validation against normalized schemas
+----------------------------+
     |
     v
+-- Layer 1: Core -----------+
|   zoo::core::Model          |   Direct llama.cpp wrapper
+----------------------------+
```

## Component Responsibilities

### Layer 1: Core (`zoo::core`)

**Model** is the direct llama.cpp wrapper. It directly owns `llama_model`, `llama_context`, `llama_sampler`, and `llama_vocab`, and manages model loading, tokenization, inference, prompt formatting (incremental via `llama_chat_apply_template()`), KV cache state, conversation history, and GPU/Metal acceleration. It is usable standalone without tools or agents.

There is no IBackend abstraction -- Model IS the llama.cpp wrapper, keeping the architecture simple and honest. The header (`model.hpp`) uses forward declarations for llama types so consumers don't need to include `llama.h`.

### Layer 2: Tools (`zoo::tools`)

| Component | Responsibility |
|-----------|---------------|
| **ToolRegistry** | Stores normalized tool definitions. Supports typed registration and manual-schema registration with deterministic ordering. Thread-safe via shared mutex. |
| **ToolCallParser** | Scans model output for JSON objects with `name` and `arguments` fields. Handles nested JSON and string escaping. |
| **ToolArgumentsValidator** | Validates parsed tool arguments against the normalized registered schema, including enum checks and unknown-argument rejection. |

The tools layer is header-only and has zero dependency on Layer 1. It operates entirely on strings, JSON, and normalized tool metadata.

### Layer 3: Agent (`zoo::Agent`)

**Agent** is the async orchestration layer. It composes a `tools::ToolRegistry` plus a private backend adapter around `core::Model`, spawns an inference thread, and implements the agentic tool loop.

Created via the `Agent::create()` factory method, which validates config, loads the model, wraps it in the private agent backend seam, and starts the inference thread. Agent is non-copyable and non-movable because the inference thread captures `this`.

The inference thread owns all model access. Cross-thread operations such as `set_system_prompt()`, `get_history()`, `clear_history()`, and tool-grammar refresh are routed through typed control commands and applied between requests.

The agentic tool loop runs inline in `process_request()`:
1. Add user message to history
2. Generate response via the private backend seam
3. Parse output for tool calls
4. If a tool call is found: validate args, record a `ToolInvocation`, execute the handler when valid, inject the tool message, loop back to step 2
5. If no tool call: return final response
6. Loop limit: 5 iterations (returns `ToolLoopLimitReached` if exceeded)

## Threading Model

Zoo-Keeper uses a two-thread architecture:

| Thread | Responsibilities |
|--------|-----------------|
| **Calling Thread** | Submits `chat()` requests, receives `std::future<Response>`, registers tools, and enqueues synchronous control commands |
| **Inference Thread** | Owns model state, processes the mailbox, runs inference, executes tools, manages history, and fires callbacks |

### Synchronization

| Resource | Mechanism |
|----------|-----------|
| Runtime mailbox | Mutex + condition variable |
| Model Access | Inference-thread ownership with typed control commands and atomic snapshots |
| Cancellation Flag | `std::atomic<bool>` per request |
| Tool Registry | `std::shared_mutex` (read-heavy) |
| Prompt/Grammar Mode | `std::atomic<bool>` snapshot published by the inference thread |

### Callback Context

All callbacks (`on_token`, tool handlers) execute on the **inference thread**. This avoids context-switching overhead during streaming but means consumers must handle their own cross-thread dispatch if updating UI or shared state.

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Three clean layers | Each layer can be used independently; strict dependency direction |
| C++23 `std::expected` | Modern, composable error handling without exceptions |
| Direct wrapping | Model owns llama.cpp resources directly -- no unnecessary abstraction |
| Value semantics | Predictable ownership, fewer lifetime bugs |
| Synchronous core | Model is single-threaded; async behavior is layered on top by Agent |
| Pure logic testing | Unit tests cover types, tools, runtime primitives, and fake-backend runtime orchestration; live Model/Agent coverage remains smoke-level integration |

## CMake Targets

| Target | Type | Links | Description |
|--------|------|-------|-------------|
| `zoo` | STATIC | llama (private, build-tree only) | Primary public runtime target containing the Model and Agent implementations |
| `zoo_core` | INTERFACE | zoo | Compatibility target that forwards to `zoo` for existing consumers |

`ZooKeeper::zoo` is the recommended link target for consumers and is available in all consumption modes (FetchContent, subdirectory, installed package). `ZooKeeper::zoo_core` is a compatibility-only alias that forwards to `zoo` — it exists for existing consumers but should not be used in new projects.

For installed-package consumers, the `ZooKeeperConfig.cmake` file locates `llama` and `nlohmann_json` via `find_dependency(...)` and attaches them to the imported target automatically. `llama` is installed alongside Zoo-Keeper in the same prefix; `nlohmann_json` must be discoverable separately because Zoo-Keeper no longer copies its headers into the install tree.

## See Also

- [Getting Started](getting-started.md) -- basic Agent setup
- [Tools](tools.md) -- tool system details
- [Building](building.md) -- build system and dependencies
