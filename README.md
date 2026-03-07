# Zoo-Keeper

[![Tests](https://img.shields.io/badge/tests-54%20passing-success)]() [![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

A C++23 library built on [llama.cpp](https://github.com/ggerganov/llama.cpp) that wraps and harnesses local LLMs for agentic behavior. Zoo-Keeper handles model loading, inference, conversation management, type-safe tool calling, and an async agentic loop -- so you can focus on your application.

## Features

- **Async Inference** -- non-blocking `chat()` with `std::future`, streaming token callbacks
- **Tool Calling** -- type-safe registration with automatic JSON schema generation
- **Agentic Loop** -- tool detection, argument validation, execution, result injection, retry
- **Context Management** -- automatic history tracking, system prompt preservation
- **Hardware Acceleration** -- Metal (macOS) and CUDA via llama.cpp
- **Modern Error Handling** -- C++23 `std::expected` throughout, no exceptions

## Quick Start

```cpp
#include <zoo/zoo.hpp>

int add(int a, int b) { return a + b; }

int main() {
    zoo::Config config;
    config.model_path = "models/llama-3-8b.gguf";
    config.context_size = 8192;

    auto agent = std::move(*zoo::Agent::create(config));
    agent->set_system_prompt("You are a helpful assistant.");
    agent->register_tool("add", "Add two numbers", {"a", "b"}, add);

    auto handle = agent->chat(zoo::Message::user("What is 42 + 58?"));
    auto response = handle.future.get();
    if (response) {
        std::cout << response->text << std::endl;
    }
}
```

## Building

```bash
git clone --recurse-submodules https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

See [docs/building.md](docs/building.md) for platform setup (Metal, CUDA), CMake options, sanitizers, coverage, and integration instructions.

## Architecture

Zoo-Keeper uses a three-layer design with strict dependency direction:

```
Layer 3: zoo::Agent        -- async orchestration, request queue, agentic tool loop
Layer 2: zoo::tools        -- tool registry, parser, validation (no llama.cpp dependency)
Layer 1: zoo::core         -- synchronous llama.cpp wrapper (Model, IBackend)
```

See [docs/architecture.md](docs/architecture.md) for the full design.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Prerequisites, build, hello-world agent, core API overview |
| [Architecture](docs/architecture.md) | Three-layer design, threading model, design principles |
| [Tools](docs/tools.md) | Template registration, supported types, manual schema, error recovery |
| [Configuration](docs/configuration.md) | Config fields, sampling params, generation limits |
| [Examples](docs/examples.md) | Streaming, tools, error handling, cancellation, metrics |
| [Building](docs/building.md) | CMake options, platform setup, sanitizers, coverage |

## Testing

54 unit tests (pure logic) using GoogleTest:

```bash
ctest --test-dir build --output-on-failure
```

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann

## License

MIT
