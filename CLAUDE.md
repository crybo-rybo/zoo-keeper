# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zoo-Keeper is a header-only C++17 library built on top of llama.cpp that functions as a complete Agent Engine for local LLM inference. It abstracts agentic AI systems by providing automated conversation history management, type-safe tool registration, asynchronous inference with cancellation support, and intelligent context window management.

**Status:** Phase 3 (MCP Client) Complete

## Git Workflow

**IMPORTANT:** Now that the initial MVP implementation is complete, all code changes MUST follow this workflow:

1. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** on the feature branch

3. **Commit your changes** with descriptive messages:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push the branch** to remote:
   ```bash
   git push -u origin feature/your-feature-name
   ```

5. **Open a Pull Request** to merge back into `main`
   - Include a clear description of changes
   - Reference any related issues
   - Ensure all tests pass
   - Request review if needed

6. **Merge via Pull Request** - DO NOT push directly to `main`

**Never commit directly to main** after the initial MVP implementation. This ensures:
- Code review process is followed
- CI/CD checks run before merging
- Clean git history with meaningful PR descriptions
- Easy rollback if needed

## Build Commands

```bash
# Configure (from project root)
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON

# Build
cmake --build build

# Run all tests
ctest --test-dir build

# Run tests with sanitizers
cmake -B build -DZOO_ENABLE_SANITIZERS=ON && cmake --build build && ctest --test-dir build

# Run with coverage
cmake -B build -DZOO_ENABLE_COVERAGE=ON && cmake --build build && ctest --test-dir build

# Build with MCP client support
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_ENABLE_MCP=ON && cmake --build build
```

### CMake Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Metal acceleration (macOS) | ON (macOS) |
| `ZOO_ENABLE_CUDA` | CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_EXAMPLES` | Build examples | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_MCP` | MCP client support | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan/TSan/UBSan | OFF |

## Architecture

### Three-Layer Design

1. **Public API Layer** - Single entry point via `zoo::Agent` class
2. **Engine Layer** - Core logic components (Request Queue, History Manager, Tool Registry, Tool Call Parser, Error Recovery, Agentic Loop)
3. **Backend Layer** - Abstracted llama.cpp interface with production and mock implementations

### Threading Model

- **Calling Thread**: Submits `chat()` requests, receives `std::future<Response>`
- **Inference Thread**: Owns `llama_context`, processes queue, executes tools, manages history

All callbacks (`on_token`, `on_tool_call`) execute on the inference thread. Consumer is responsible for cross-thread synchronization.

### Key Design Decisions

- **Header-only distribution**: Simplifies integration
- **`std::expected` for errors**: Modern composable error handling without exceptions
- **Dependency injection for llama.cpp**: Enables unit testing via Mock Backend
- **Value semantics**: Predictable ownership, fewer lifetime bugs

## Test Structure

```
tests/
├── unit/
│   ├── test_request_queue.cpp
│   ├── test_history_manager.cpp
│   ├── test_tool_registry.cpp
│   ├── test_agentic_loop.cpp
│   ├── test_error_recovery.cpp
│   └── test_agent.cpp
├── mocks/
│   ├── mock_backend.hpp
│   └── mock_backend.cpp
├── fixtures/
│   ├── sample_responses.hpp
│   ├── tool_definitions.hpp
│   └── template_expectations.hpp
└── CMakeLists.txt
```

Uses GoogleTest/GoogleMock. Follow TDD: Red (failing test) → Green (minimal code to pass) → Refactor.

## Dependencies

| Dependency | Integration |
|------------|-------------|
| llama.cpp | Git submodule |
| nlohmann/json | CMake FetchContent |
| GoogleTest 1.14+ | CMake FetchContent (tests only) |

## Compiler Requirements

- **Windows**: MSVC 2019+ (19.20)
- **macOS**: Clang 13.0+
- **Linux**: GCC 11.0+ or Clang 13.0+

## Key Types

- `zoo::Agent` - Main entry point, owns inference thread
- `zoo::Config` - Immutable configuration (model path, context size, sampling params, template)
- `zoo::Message` - Value type with role (System/User/Assistant/Tool) and content
- `zoo::Response` - Result containing text, tool history, token usage, latency metrics
- `zoo::ToolCall` - Represents a parsed tool call with id, name, and arguments
- `zoo::engine::ToolEntry` - Metadata and handler for a registered tool
- `zoo::engine::ToolRegistry` - Thread-safe registry for tool definitions and invocation
- `zoo::engine::ToolCallParser` - Detects and extracts tool calls from model output
- `zoo::engine::ErrorRecovery` - Validates tool arguments and tracks retry attempts
- `zoo::mcp::McpClient` - Connects to an MCP server, discovers tools, wraps them into ToolRegistry
- `zoo::mcp::protocol::Session` - Manages JSON-RPC session lifecycle with an MCP server
- `zoo::mcp::protocol::MessageRouter` - Routes JSON-RPC responses/notifications to pending requests
- `zoo::mcp::protocol::JsonRpc` - JSON-RPC 2.0 message serialization and deserialization

## Tool Registration

Template-based registration supporting `int`, `float`, `double`, `bool`, `std::string` parameters:
```cpp
// Function signature extraction example
int add(int a, int b) { return a + b; }

// Registration with parameter names
agent.register_tool("add", "Adds two numbers", {"a", "b"}, add);
```

Generates JSON schema automatically from function signature. The `param_names` vector must match the function's arity.

## Error Codes (Phase 2 Additions)

New error codes for the tool system:
- `ErrorCode::ToolNotFound` (500) - Requested tool not found in registry
- `ErrorCode::ToolExecutionFailed` (501) - Tool handler threw exception or returned error
- `ErrorCode::InvalidToolSignature` (502) - Tool signature does not match supported types
- `ErrorCode::ToolRetriesExhausted` (503) - Maximum retry attempts exceeded for tool
- `ErrorCode::ToolLoopLimitReached` (504) - Maximum tool loop iterations exceeded

## Error Codes (Phase 3 Additions)

New error codes for the MCP client:
- `ErrorCode::McpTransportFailed` (600) - Transport layer failed (process spawn, pipe I/O)
- `ErrorCode::McpProtocolError` (601) - JSON-RPC protocol violation
- `ErrorCode::McpServerError` (602) - MCP server returned an error response
- `ErrorCode::McpSessionFailed` (603) - Session initialization or capability negotiation failed
- `ErrorCode::McpToolNotAvailable` (604) - Requested tool not available on the MCP server
- `ErrorCode::McpTimeout` (605) - Request to MCP server timed out
- `ErrorCode::McpDisconnected` (606) - MCP server disconnected unexpectedly

## Documentation

- `docs/getting-started.md` - Prerequisites, build, hello-world agent, core API
- `docs/tools.md` - Tool registration, supported types, error recovery
- `docs/context-database.md` - SQLite long-term memory, pruning, FTS5 retrieval
- `docs/rag.md` - IRetriever interface, InMemoryRagStore, ephemeral injection
- `docs/architecture.md` - Three-layer design, threading model, design principles
- `docs/configuration.md` - Config fields, sampling params, templates, ChatOptions
- `docs/examples.md` - Streaming, tools, RAG, context DB, error handling, cancellation
- `docs/mcp.md` - MCP client, tool federation, transport, error codes
- `docs/building.md` - CMake options, platform setup, sanitizers, coverage
