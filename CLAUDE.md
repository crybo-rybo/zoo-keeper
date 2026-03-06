# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zoo-Keeper is a C++23 library built on top of llama.cpp that serves as a llama.cpp wrapper/harness enabling agentic behavior from locally hosted LLMs. It provides model loading, inference, conversation management, type-safe tool registration, and an async agentic loop.

**Platforms:** Linux and macOS only.

## Git Workflow

All code changes MUST follow this workflow:

1. Create a feature branch from `main`
2. Make changes on the feature branch
3. Push the branch and open a Pull Request
4. Merge via Pull Request - DO NOT push directly to `main`

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
```

### CMake Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Metal acceleration (macOS) | ON (macOS) |
| `ZOO_ENABLE_CUDA` | CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_EXAMPLES` | Build examples | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan/TSan/UBSan | OFF |

## Architecture — Three Clean Layers

```
Layer 3: zoo::Agent (agent.hpp)
  Async orchestration: inference thread, request queue, agentic tool loop
  Composes Layer 1 + Layer 2

Layer 2: zoo::tools (tools/)
  Tool registry, tool call parsing, argument validation
  Standalone — no dependency on Layer 1

Layer 1: zoo::core (core/)
  Model loading, inference, tokenization, chat formatting, KV cache, history
  Zero dependency on Layers 2 or 3
```

### Threading Model

- **Calling Thread**: Submits `chat()` requests, receives `std::future<Response>`
- **Inference Thread**: Owns Model, processes queue, executes tools, manages history

All callbacks execute on the inference thread. Consumer is responsible for cross-thread synchronization.

### Key Design Decisions

- **C++23**: Uses `std::expected` (no external expected library)
- **`zoo::core::Model`**: Synchronous, single-threaded llama.cpp wrapper. Usable standalone.
- **Dependency injection**: `core::IBackend` interface enables testing with MockBackend
- **Value semantics**: Predictable ownership, fewer lifetime bugs
- **Header-only tools layer**: Tools have no llama.cpp dependency

### CMake Targets

| Target | Description | Links |
|--------|-------------|-------|
| `zoo` | INTERFACE library (headers + nlohmann_json) | — |
| `zoo_model` | STATIC library (Model class, no llama dep) | zoo |
| `zoo_backend` | STATIC library (LlamaBackend) | zoo_model, llama |

## File Structure

```
include/zoo/
├── core/
│   ├── types.hpp          # Message, Role, Error, Config, SamplingParams, Response
│   └── model.hpp          # Model class + IBackend interface
├── tools/
│   ├── types.hpp          # ToolCall, ToolEntry, ToolHandler
│   ├── registry.hpp       # ToolRegistry (template registration + invocation)
│   ├── parser.hpp         # ToolCallParser
│   └── validation.hpp     # ErrorRecovery
├── agent.hpp              # Agent class + RequestQueue + RequestHandle
└── zoo.hpp                # Convenience header
src/core/
├── model.cpp              # Model implementation (no llama dep)
└── llama_backend.cpp      # LlamaBackend implementation
tests/
├── unit/
│   ├── test_types.cpp
│   ├── test_model.cpp
│   ├── test_tool_registry.cpp
│   ├── test_tool_parser.cpp
│   ├── test_error_recovery.cpp
│   └── test_agent.cpp
├── mocks/
│   ├── mock_backend.hpp
│   └── mock_backend.cpp
└── fixtures/
    ├── sample_responses.hpp
    └── tool_definitions.hpp
```

## Dependencies

| Dependency | Integration |
|------------|-------------|
| llama.cpp | Git submodule (`extern/llama.cpp`) |
| nlohmann/json | CMake FetchContent |
| GoogleTest 1.14+ | CMake FetchContent (tests only) |

## Compiler Requirements

- **macOS**: Clang 16.0+ (C++23 support)
- **Linux**: GCC 13.0+ or Clang 16.0+

## Key Types

- `zoo::Config` — Model path, context size, sampling params, max tokens
- `zoo::Message` — Value type with role (System/User/Assistant/Tool) and content
- `zoo::Response` — Generated text, token usage, metrics, tool call history
- `zoo::core::Model` — Synchronous llama.cpp wrapper (Layer 1)
- `zoo::core::IBackend` — Backend interface for dependency injection
- `zoo::tools::ToolCall` — Parsed tool call with id, name, arguments
- `zoo::tools::ToolRegistry` — Thread-safe tool registration and invocation
- `zoo::tools::ToolCallParser` — Detects tool calls in model output
- `zoo::tools::ErrorRecovery` — Argument validation and retry tracking
- `zoo::Agent` — Async orchestrator with agentic tool loop (Layer 3)

## Tool Registration

```cpp
int add(int a, int b) { return a + b; }
agent.register_tool("add", "Adds two numbers", {"a", "b"}, add);
```

Supported types: `int`, `float`, `double`, `bool`, `std::string`. JSON schema generated automatically from function signature.

## Error Codes

- 100-199: Configuration errors
- 200-299: Backend/model errors
- 300-399: Engine logic errors
- 400-499: Runtime/request errors
- 500-599: Tool system errors
