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

Zoo-Keeper is a local-first C++23 library for building serious LLM-backed applications without hand-rolling the hard parts every time. It wraps [llama.cpp](https://github.com/ggerganov/llama.cpp) with a direct synchronous model API, then layers an async agent runtime on top for queued requests, streaming, tool execution, and conversation management.

The goal is not to be a giant framework. The goal is to give you a small number of explicit, composable primitives that feel natural in C++: a `zoo::core::Model` when you want direct control, a `zoo::Agent` when you want orchestration, and a tool system that turns native callables into model-usable capabilities with schema generation, validation, and execution tracking.

## Why Zoo-Keeper

- **Local-first by default**: run on your own hardware through `llama.cpp`, with Metal on macOS and CUDA support when enabled.
- **Two clear abstraction levels**: use the low-level `Model` directly or move up to `Agent` when you need async orchestration.
- **Native C++ tools**: register regular functions or manual JSON-backed handlers and let the library handle schema generation, validation, and execution bookkeeping.
- **Predictable concurrency**: model access stays on the inference thread, while callers interact through request handles and futures.
- **Modern contracts**: `std::expected`-based error handling, explicit response metadata, and a smaller public target story centered on `ZooKeeper::zoo`.

## Choose Your Surface

| Surface | Use it when you need | What it gives you |
|--------|-----------------------|-------------------|
| `zoo::core::Model` | Direct, single-threaded control | Model loading, tokenization, generation, history, KV-cache management |
| `zoo::Agent` | Async chat, stateless completion, streaming, and tool orchestration | Request queue, inference thread, cancellation, tool loop, `RequestHandle`, structured tool invocation tracking |

## Quick Start

```cpp
#include <iostream>
#include <zoo/zoo.hpp>

int add(int a, int b) {
    return a + b;
}

int main() {
    zoo::Config config;
    config.model_path = "models/llama-3-8b.gguf";
    config.context_size = 8192;
    config.max_tokens = 512;
    config.n_gpu_layers = 0;

    auto agent_result = zoo::Agent::create(config);
    if (!agent_result) {
        std::cerr << agent_result.error().to_string() << '\n';
        return 1;
    }

    auto agent = std::move(*agent_result);
    agent->set_system_prompt("You are a concise assistant. Use tools when helpful.");

    auto tool_result = agent->register_tool("add", "Add two integers", {"a", "b"}, add);
    if (!tool_result) {
        std::cerr << tool_result.error().to_string() << '\n';
        return 1;
    }

    auto handle = agent->chat(zoo::Message::user("What is 42 + 58?"));
    auto response = handle.future.get();

    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
```

## What You Get

- **Async inference** with per-token streaming callbacks and cancellable requests
- **Structured tool calling** with typed registration, manual schema registration, validation, and explicit `tool_invocations`
- **Grammar-constrained structured output** via `extract()` — enforce a JSON Schema at the sampler level with no retry loops
- **Conversation management** with system prompts, history tracking, and incremental prompt/KV-cache handling
- **Deterministic tool metadata** for schema and grammar generation
- **Concrete response metadata** including token usage and latency metrics

## Build and Link

```bash
git clone --recurse-submodules https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
scripts/build -DZOO_BUILD_EXAMPLES=ON
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
| [Migration](MIGRATION.md) | Consumer-facing changes to move from `0.2.x` to `1.0` |
| [API Reference](docs/building.md#api-reference) | Generate Doxygen locally or browse the published reference |

Generate the API reference locally with:

```bash
cmake --preset docs && cmake --build --preset docs
```

## Testing

The default build runs the full unit suite, and CI also smoke-tests build-tree and installed-package CMake consumers.

```bash
scripts/test
```

For live smoke coverage against a real model:

```bash
scripts/build -DZOO_BUILD_INTEGRATION_TESTS=ON
ZOO_INTEGRATION_MODEL=/absolute/path/to/model.gguf scripts/test
```

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann
- [GoogleTest](https://github.com/google/googletest) by Google

## License

MIT
