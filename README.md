<p align="center">
  <img src="docs/images/zoo_keeper_logo.png" alt="Zoo-Keeper logo" width="220">
</p>

<h1 align="center">Zoo-Keeper</h1>

<p align="center">
  <b>Build local, tool-using LLM applications in modern C++.</b><br/>
  <sub>llama.cpp-backed &bull; Async agent runtime &bull; Type-safe native tools</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-23-blue" alt="C++23" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/tests-ctest%20passing-success" alt="Tests" />
</p>

## About

Zoo-Keeper is a local-first C++23 library for building LLM-backed applications on top of [llama.cpp](https://github.com/ggerganov/llama.cpp). It keeps the public surface small and explicit: `zoo::core::Model` for direct inference, `zoo::Agent` for async orchestration, and `zoo::tools` for turning native callables into model-usable tools.

The release API is split into `zoo::ModelConfig`, `zoo::AgentConfig`, and `zoo::GenerationOptions`. Async calls return `RequestHandle<Result>` objects with `id()`, `ready()`, and `await_result()`, while completed text and extraction calls come back as `TextResponse` and `ExtractionResponse`.

## Why Zoo-Keeper

- **Local-first by default**: run on your own hardware through `llama.cpp`, with Metal on macOS and CUDA support when enabled.
- **Two clear abstraction levels**: use the low-level `Model` directly or move up to `Agent` when you need queued requests and tool loops.
- **Native C++ tools**: register regular functions or JSON-backed handlers and let the library handle schema generation, validation, and execution tracking.
- **Predictable concurrency**: callers interact through request handles, not raw futures, and the inference thread owns model state.
- **Modern contracts**: `std::expected`-based error handling, borrowed message views, owned history snapshots, and explicit response metadata.

## Choose Your Surface

| Surface | Use it when you need | What it gives you |
|--------|-----------------------|-------------------|
| `zoo::core::Model` | Direct, single-threaded control | Model loading, generation, retained history, and KV-cache management |
| `zoo::Agent` | Async chat, stateless completion, streaming, and tool orchestration | Request queue, inference thread, cancellation, tool loop, and `RequestHandle<Result>` |
| `zoo::ModelConfig` / `zoo::AgentConfig` / `zoo::GenerationOptions` | Configure runtime startup and per-call generation | Split config blocks with strict validation and JSON helpers |
| `zoo::MessageView` / `zoo::ConversationView` / `zoo::HistorySnapshot` | Pass or inspect conversation state | Borrowed request-scoped views and owning history snapshots |

## Quick Start

```cpp
#include <iostream>
#include <zoo/zoo.hpp>

int main() {
    zoo::ModelConfig model;
    model.model_path = "models/llama-3-8b.gguf";
    model.context_size = 8192;
    model.n_gpu_layers = 0;

    zoo::AgentConfig agent;
    agent.max_history_messages = 32;

    zoo::GenerationOptions generation;
    generation.max_tokens = 256;

    auto agent_result = zoo::Agent::create(model, agent, generation);
    if (!agent_result) {
        std::cerr << agent_result.error().to_string() << '\n';
        return 1;
    }

    auto agent = std::move(*agent_result);
    agent->set_system_prompt("You are a concise assistant. Use tools when helpful.");

    auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "What is 42 + 58?"});
    auto response = handle.await_result();

    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
```

## What You Get

- **Async inference** with per-token streaming callbacks, cancellable requests, and `RequestHandle<Result>`
- **Structured tool calling** with typed registration, manual JSON schema registration, validation, and optional `tool_trace`
- **Grammar-constrained structured output** via `extract()` and `ExtractionResponse::data`
- **Conversation management** with system prompts, `MessageView`/`ConversationView`, and `HistorySnapshot`
- **Concrete response metadata** including token usage, latency, and throughput metrics

## Build and Link

```bash
git clone --recurse-submodules https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

For consumers, `ZooKeeper::zoo` is the primary CMake target. See [docs/building.md](docs/building.md) for:

- submodule and `FetchContent` setup
- installed-package usage
- Metal and CUDA options
- sanitizers and coverage
- integration-test setup

## Architecture at a Glance

Zoo-Keeper follows a strict three-layer design:

```text
Layer 3: zoo::Agent  -> async orchestration, request queue, streaming, tool loop
Layer 2: zoo::tools  -> registry, parser, schema validation, grammar generation
Layer 1: zoo::core   -> direct llama.cpp wrapper and incremental prompt management
```

This split keeps the synchronous llama.cpp wrapper honest and small, while the agent runtime owns concurrency and orchestration explicitly instead of hiding it behind a vague abstraction.

See [docs/architecture.md](docs/architecture.md) for the full threading model and internal boundaries.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | First build, first agent, and core API overview |
| [Building](docs/building.md) | CMake usage, platform setup, sanitizers, coverage, install/package details |
| [Configuration](docs/configuration.md) | Runtime config, sampling parameters, generation limits, and history budgets |
| [Tools](docs/tools.md) | Typed tools, manual schema registration, supported schema subset, and error handling |
| [Structured Output](docs/extract.md) | Grammar-constrained extraction, schema reference, stateful vs. stateless, error codes |
| [Examples](docs/examples.md) | Focused example programs for streaming, cancellation, tools, and error handling |
| [Architecture](docs/architecture.md) | Layering, runtime ownership, threading model, and target structure |
| [Compatibility](docs/compatibility.md) | Public API boundary, intended 1.x stability policy, and deprecation rules |
| [Migration](MIGRATION.md) | Upgrade notes for major API changes, including `v1.0.3` to `v1.1.0` |
| [API Reference](docs/building.md#api-reference) | Generate Doxygen locally or browse the published reference |

Generate the API reference locally with:

```bash
cmake --preset docs && cmake --build --preset docs
```

## Testing

The default build runs the test suite, and CI also smoke-tests build-tree and installed-package CMake consumers.

```bash
scripts/test.sh
```

For live smoke coverage against a real model:

```bash
scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON
ZOO_INTEGRATION_MODEL=/absolute/path/to/model.gguf scripts/test.sh
```

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann
- [GoogleTest](https://github.com/google/googletest) by Google

## License

MIT
