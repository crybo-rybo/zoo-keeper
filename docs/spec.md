# spec.md — Zoo-Keeper

## Objective

A local-first C++23 library that wraps llama.cpp to provide model loading,
inference, conversation management, type-safe tool registration, and an async
agentic loop for locally hosted LLMs.

Zoo-Keeper gives C++ developers a small number of explicit, composable
primitives: `zoo::core::Model` for direct synchronous control, `zoo::Agent`
for async orchestration, and a tool system that turns native callables into
model-usable capabilities.

## Tech Stack

- **Language:** C++23 (GCC 13+, Clang 18+, Apple Clang 16+)
- **Build system:** CMake 3.18+
- **Runtime dependency:** llama.cpp (git submodule at `extern/llama.cpp`)
- **JSON:** nlohmann/json (CMake FetchContent)
- **Test framework:** GoogleTest 1.14+ (CMake FetchContent, test-only)

## Supported Platforms

- Linux (GCC 13+ or Clang 18+)
- macOS (Apple Clang 16+, Metal acceleration)

## Public API Boundary

The supported public surface is:

- Installed headers under `include/zoo/` (excluding `include/zoo/internal/`)
- CMake target `ZooKeeper::zoo` (primary) and `ZooKeeper::zoo_core` (compat)
- Types: `zoo::Agent`, `zoo::core::Model`, `zoo::Config`, `zoo::Message`,
  `zoo::Response`, `zoo::tools::ToolRegistry`, `zoo::tools::ToolCallParser`,
  `zoo::tools::ErrorRecovery`

Everything under `include/zoo/internal/`, `src/`, and private CMake plumbing
is **not** part of the compatibility contract.

## Architecture

Three strict layers — each depends only on layers below it:

| Layer | Namespace | Responsibility |
|-------|-----------|----------------|
| 3 | `zoo::Agent` | Async orchestration, inference thread, request queue, tool loop |
| 2 | `zoo::tools` | Tool registry, call parsing, schema validation, grammar. Header-only, zero llama.cpp dep |
| 1 | `zoo::core` | Direct synchronous llama.cpp wrapper. Owns all llama resources |

## Error Handling

`std::expected<T, zoo::Error>` everywhere. No exceptions. Error codes by category:

- 100–199: Configuration
- 200–299: Backend/model
- 300–399: Engine logic
- 400–499: Runtime/request
- 500–599: Tool system

## Non-Goals

- Windows support
- Multi-backend abstraction (Zoo-Keeper targets llama.cpp only)
- HTTP/REST API server (this is a library, not a service)
- Python bindings
- Distributed inference

## Verification Rules

- All unit tests pass: `scripts/test`
- Formatting produces no diff: `scripts/format`
- Builds are warning-free: `scripts/lint`
- Public API changes require review

## Done When

A change is complete when:

1. Behavior is observable (not just "compiles")
2. `scripts/test` passes (all 219+ unit tests)
3. `scripts/format` produces no diff
4. Build is warning-free
5. Integration tests pass if model-touching code changed
