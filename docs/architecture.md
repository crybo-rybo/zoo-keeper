# Architecture

Zoo-Keeper follows a three-layer architecture that separates concerns between the public API, core engine logic, and inference backend.

## Layer Structure

```
Consumer Code
     |
     v
+-- Public API Layer -------+
|   zoo::Agent               |   Single entry point for all functionality
+----------------------------+
     |
     v
+-- Engine Layer ------------+
|   RequestQueue              |   Thread-safe MPSC request queue
|   HistoryManager            |   Conversation state + context window management
|   ToolRegistry              |   Tool definitions, schema generation, invocation
|   ToolCallParser            |   Tool call detection in model output
|   ErrorRecovery             |   Argument validation + retry tracking
|   AgenticLoop               |   Inference -> tool -> inject -> loop orchestration
|   McpClient                 |   MCP server connection, tool discovery & federation
|   Session / MessageRouter   |   JSON-RPC lifecycle and response routing
|   JsonRpc                   |   JSON-RPC 2.0 serialization
|   ITransport / StdioTransport|  Abstract transport; production stdio pipe impl
+----------------------------+
     |
     v
+-- Backend Layer -----------+
|   IBackend (interface)      |   Abstract inference operations
|   LlamaBackend              |   Production: wraps llama.cpp
|   MockBackend               |   Testing: scripted responses
+----------------------------+
```

## Component Responsibilities

### Public API Layer

**Agent** is the single entry point. It owns all internal components, spawns the inference thread, and exposes a thread-safe public API. Created via the `Agent::create()` factory method, which validates config, initializes the backend, and starts the inference thread.

Agent is non-copyable and non-movable because the inference thread captures `this`.

### Engine Layer

| Component | Responsibility |
|-----------|---------------|
| **RequestQueue** | Thread-safe queue connecting calling threads to the inference thread. Multiple producers, single consumer. |
| **HistoryManager** | Owns the `vector<Message>` conversation history. Validates role sequences, tracks estimated token counts, implements FIFO pruning with system prompt preservation. |
| **ToolRegistry** | Stores tool definitions with JSON schemas. Supports template-based registration (auto-generates schema from function signatures) and manual registration. Thread-safe via shared mutex. |
| **ToolCallParser** | Scans model output for JSON objects with `name` and `arguments` fields. Handles nested JSON and string escaping. |
| **ErrorRecovery** | Validates tool call arguments against registered schemas. Tracks retry counts per tool with configurable limits (default: 2). |
| **AgenticLoop** | The core orchestration loop. Formats prompts, runs inference, detects tool calls, validates/executes/injects results, and loops until a final response or limit is reached. Also handles RAG context injection and context database pruning. |

### Backend Layer

**IBackend** defines the abstract interface that decouples the engine from llama.cpp:

- `initialize(config)` -- load model
- `format_prompt(messages)` -- incremental prompt building
- `tokenize(text)` -- text to tokens
- `generate(tokens, max_tokens, stop_sequences, callback)` -- run inference
- `finalize_response(messages)` -- update prompt cache state
- `clear_kv_cache()` -- reset KV cache

**LlamaBackend** is the production implementation. It owns `llama_model` and `llama_context`, manages formatting state (`prev_len_`, `formatted_`) for incremental prompt building, and handles GPU/Metal acceleration.

**MockBackend** supports scripted responses, tool call simulation, and error injection for unit testing without real models.

## Threading Model

Zoo-Keeper uses a two-thread architecture:

| Thread | Responsibilities |
|--------|-----------------|
| **Calling Thread** | Submits `chat()` requests, receives `std::future<Response>`, registers tools, sets system prompt |
| **Inference Thread** | Processes request queue, runs inference, executes tools, manages history, fires callbacks |
| **MCP Transport Thread** (per server) | Reads stdout from MCP server process, routes responses via MessageRouter |

### Synchronization

| Resource | Mechanism |
|----------|-----------|
| Request Queue | Mutex + condition variable |
| Cancellation Flag | `std::atomic<bool>` |
| System Prompt / History | `std::mutex` (shared between Agent and AgenticLoop) |
| Tool Registry | `std::shared_mutex` (read-heavy) |

### Callback Context

All callbacks (`on_token`, tool handlers) execute on the **inference thread**. This avoids context-switching overhead during streaming but means consumers must handle their own cross-thread dispatch if updating UI or shared state.

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Header-only distribution | Simplifies integration; no linking complexity |
| Single `Agent` entry point | Minimal API surface; easy to learn |
| `std::expected` for errors | Modern, composable error handling without exceptions |
| Dependency injection for backend | Enables comprehensive unit testing via MockBackend |
| Value semantics | Predictable ownership, fewer lifetime bugs |

## See Also

- [Getting Started](getting-started.md) -- basic Agent setup
- [Tools](tools.md) -- tool system details
- [Building](building.md) -- build system and dependencies
