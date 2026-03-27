# Zoo-Keeper Architecture Snapshot

This file is a current-state reference, not release guidance. If it drifts from HEAD,
trust the public headers, examples, and changelog first.

## Objective

Zoo-Keeper is a local-first C++23 library on top of llama.cpp. It provides model
loading, inference, conversation management, native tool calling, structured
extraction, and an async agent loop for locally hosted LLMs.

The public API is intentionally small and explicit: `zoo::core::Model` for low-level
control, `zoo::Agent` for async orchestration, and split config types for model,
agent, and per-call generation policy.

## Tech Stack

- C++23
- CMake
- llama.cpp as a vendored submodule
- nlohmann/json for config and schema handling
- GoogleTest for unit and integration tests

## Supported Platforms

- Linux
- macOS

## Public API Boundary

The supported surface is:

- Installed headers under `include/zoo/`, excluding `include/zoo/internal/`
- CMake target `ZooKeeper::zoo`
- Core types and APIs such as `ModelConfig`, `AgentConfig`, `GenerationOptions`,
  `zoo::Agent`, `zoo::core::Model`, `Message`, `MessageView`, `ConversationView`,
  `HistorySnapshot`, `TextResponse`, `ExtractionResponse`, `RequestHandle<T>`,
  `ToolRegistry`, `ToolCallParser`, and `ErrorRecovery`

Everything under `include/zoo/internal/` and `src/` is intentionally private.

## Architecture

Three strict layers, each depending only on the layers below it:

| Layer | Namespace | Responsibility |
|-------|-----------|----------------|
| 3 | `zoo::Agent` | Async orchestration, request handles, history management, tool execution |
| 2 | `zoo::tools` | Tool registry, call parsing, schema validation, grammar generation. Header-only and llama.cpp-free |
| 1 | `zoo::core` | Direct llama.cpp wrapper, prompt rendering, native tool calling, structured extraction |

The current core layer owns the model, context, sampler, chat templates, and the
template-driven tool/extraction grammar state. The agent layer chooses request shape
and preserves or restores history as needed.

## Error Handling

`std::expected<T, zoo::Error>` is used throughout the public surface. Exceptions are
not part of the public API contract.

## Non-Goals

- Windows support
- Multi-backend abstraction
- HTTP/REST service wrapper
- Python bindings
- Distributed inference

## Maintenance Rules

- Keep this document aligned with the current public headers and examples
- Prefer the changelog for release-specific summaries
- Treat `include/zoo/internal/` and `src/` as implementation details

## Verification Expectations

- `scripts/test.sh` passes
- `scripts/format.sh` produces no diff
- `scripts/lint.sh` is warning-free
- Public API changes should be reviewed deliberately
