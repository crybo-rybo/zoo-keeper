# Architecture

Zoo-Keeper exposes three public layers with a simple dependency direction: higher layers build on lower layers, and consumers can stop at the lowest layer that fits their needs.

## Public Layers

| Layer | Primary Types | Responsibility |
|-------|---------------|----------------|
| Agent | `zoo::Agent`, `zoo::RequestHandle` | Async request submission, background inference, tool loop orchestration |
| Tools | `zoo::tools::ToolRegistry`, `zoo::tools::ToolCallParser`, `zoo::tools::ToolArgumentsValidator` | Tool registration, tool-call parsing, schema validation |
| Core | `zoo::core::Model` | Direct synchronous llama.cpp wrapper |

## Usage Model

### `zoo::core::Model`

Use `Model` when you want direct, single-threaded inference without the agent runtime. It owns model loading, prompt rendering, history, KV-cache interaction, sampling, and generation.

### `zoo::Agent`

Use `Agent` when you want queued asynchronous requests, streaming callbacks, cancellation, model-driven tool execution, and a choice between stateful `chat(...)` requests and stateless request-scoped `complete(...)` requests. `Agent` is the primary high-level runtime surface for most consumers.

## Public Threading Guarantees

- `zoo::Agent` owns a background inference thread.
- Requests are submitted from the calling thread through `chat(...)` or `complete(...)` and resolved through `RequestHandle::future`.
- Model state is owned by the inference thread while the agent is running.
- Streaming callbacks and tool handlers execute on the inference thread.

These guarantees are part of the public behavioral contract. Private runtime mechanisms that implement them are documented separately for maintainers.

## CMake Targets

| Target | Status | Notes |
|--------|--------|-------|
| `ZooKeeper::zoo` | Primary | Recommended target for new consumers |
| `ZooKeeper::zoo_core` | Compatibility only | Forwarding target retained for existing consumers |

## Design Goals

- One obvious public runtime story centered on `ZooKeeper::zoo`
- Small installed API surface under `include/zoo/`
- Explicit tool execution data and deterministic tool metadata behavior
- Docs that describe supported behavior without exposing private implementation details as API

## For Maintainers

Internal runtime ownership, private module boundaries, and contributor-facing invariants live in [maintainer-architecture.md](maintainer-architecture.md).
