# Architecture

Zoo-Keeper follows a three-layer architecture with strict dependency direction. Each layer depends only on the layers below it.

## Layer Structure

```
Consumer Code
     |
     v
+-- Layer 3: Agent ---------+
|   zoo::Agent               |   Async orchestration, request queue, agentic tool loop
|   RequestQueue             |   Thread-safe MPSC request queue
|   RequestHandle            |   Future-based response handle
+----------------------------+
     |
     v
+-- Layer 2: Tools ----------+
|   zoo::tools::ToolRegistry  |   Tool definitions, schema generation, invocation
|   zoo::tools::ToolCallParser |   Tool call detection in model output
|   zoo::tools::ErrorRecovery  |   Argument validation + retry tracking
+----------------------------+
     |
     v
+-- Layer 1: Core -----------+
|   zoo::core::Model          |   Synchronous llama.cpp wrapper
|   zoo::core::IBackend       |   Abstract inference operations
|   zoo::core::LlamaBackend   |   Production: wraps llama.cpp
+----------------------------+
```

## Component Responsibilities

### Layer 1: Core (`zoo::core`)

**Model** is a synchronous, single-threaded llama.cpp wrapper. It manages model loading, conversation history, prompt formatting, tokenization, inference, and KV cache state. It is usable standalone without tools or agents.

**IBackend** defines the abstract interface that decouples the Model from llama.cpp:

- `initialize(config)` -- load model, create context
- `format_prompt(messages)` -- incremental prompt building via `llama_chat_apply_template()`
- `tokenize(text)` -- text to tokens
- `generate(tokens, max_tokens, stop_sequences, callback)` -- run inference
- `finalize_response(messages)` -- update prompt cache state
- `clear_kv_cache()` -- reset KV cache

**LlamaBackend** is the production implementation. It owns `llama_model`, `llama_context`, and `llama_sampler`, manages formatting state for incremental prompt building, and handles GPU/Metal acceleration.

The CMake target split enables testing without llama.cpp:
- `zoo_model` links only `zoo` (headers + nlohmann_json) -- tests use this
- `zoo_backend` links `zoo_model` + llama.cpp -- production code uses this

### Layer 2: Tools (`zoo::tools`)

| Component | Responsibility |
|-----------|---------------|
| **ToolRegistry** | Stores tool definitions with JSON schemas. Supports template-based registration (auto-generates schema from function signatures) and manual registration. Thread-safe via shared mutex. |
| **ToolCallParser** | Scans model output for JSON objects with `name` and `arguments` fields. Handles nested JSON and string escaping. |
| **ErrorRecovery** | Validates tool call arguments against registered schemas. Tracks retry counts per tool with configurable limits (default: 2). |

The tools layer is header-only and has zero dependency on Layer 1. It operates entirely on strings and JSON.

### Layer 3: Agent (`zoo::Agent`)

**Agent** is the async orchestration layer. It composes a `core::Model` and `tools::ToolRegistry`, spawns an inference thread, and implements the agentic tool loop.

Created via the `Agent::create()` factory method, which validates config, initializes the backend, loads the model, and starts the inference thread. Agent is non-copyable and non-movable because the inference thread captures `this`.

The agentic tool loop runs inline in `process_request()`:
1. Add user message to history
2. Generate response via `Model::generate_from_history()`
3. Parse output for tool calls
4. If tool call found: validate args, execute handler, inject result, loop back to step 2
5. If no tool call: return final response
6. Loop limit: 5 iterations (returns `ToolLoopLimitReached` if exceeded)

## Threading Model

Zoo-Keeper uses a two-thread architecture:

| Thread | Responsibilities |
|--------|-----------------|
| **Calling Thread** | Submits `chat()` requests, receives `std::future<Response>`, registers tools, sets system prompt |
| **Inference Thread** | Processes request queue, runs inference, executes tools, manages history, fires callbacks |

### Synchronization

| Resource | Mechanism |
|----------|-----------|
| Request Queue | Mutex + condition variable |
| Model Access | `std::mutex` (model_mutex_) |
| Cancellation Flag | `std::atomic<bool>` per request |
| Tool Registry | `std::shared_mutex` (read-heavy) |

### Callback Context

All callbacks (`on_token`, tool handlers) execute on the **inference thread**. This avoids context-switching overhead during streaming but means consumers must handle their own cross-thread dispatch if updating UI or shared state.

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Three clean layers | Each layer can be used independently; strict dependency direction |
| C++23 `std::expected` | Modern, composable error handling without exceptions |
| Dependency injection | `IBackend` interface enables testing via MockBackend without llama.cpp |
| Value semantics | Predictable ownership, fewer lifetime bugs |
| Synchronous core | Model is single-threaded; async behavior is layered on top by Agent |

## CMake Targets

| Target | Type | Links | Description |
|--------|------|-------|-------------|
| `zoo` | INTERFACE | nlohmann_json | Headers only (types, tools) |
| `zoo_model` | STATIC | zoo | Model class (no llama.cpp dependency) |
| `zoo_backend` | STATIC | zoo_model, llama | LlamaBackend (production inference) |

## See Also

- [Getting Started](getting-started.md) -- basic Agent setup
- [Tools](tools.md) -- tool system details
- [Building](building.md) -- build system and dependencies
