# Zoo-Keeper

[![Tests](https://img.shields.io/badge/tests-316%20passing-success)]() [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

A header-only C++17 Agent Engine for local LLM inference, built on [llama.cpp](https://github.com/ggerganov/llama.cpp). Zoo-Keeper handles the hard parts of building agentic AI -- conversation management, tool calling, context window pressure, and retrieval-augmented generation -- so you can focus on your application.

## Features

- **Async Inference** -- non-blocking `chat()` with `std::future`, streaming token callbacks
- **Tool Calling** -- type-safe registration with automatic JSON schema generation
- **Agentic Loop** -- tool detection, validation, execution, result injection, retry
- **MCP Integration** -- connect to Model Context Protocol servers for tool federation
- **RAG** -- per-request ephemeral context injection via pluggable retrievers
- **Long-Term Memory** -- SQLite context database with automatic archival and retrieval
- **Context Management** -- automatic history tracking, FIFO pruning, system prompt preservation
- **Hardware Acceleration** -- Metal (macOS) and CUDA via llama.cpp
- **Modern Error Handling** -- `std::expected` throughout, no exceptions

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

    auto response = agent->chat(zoo::Message::user("What is 42 + 58?")).get();
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

See [docs/building.md](docs/building.md) for platform setup (Metal, CUDA), CMake options, sanitizers, coverage, and FetchContent integration.

## Architecture

Zoo-Keeper uses a three-layer design:

- **Public API** -- `zoo::Agent` is the single entry point; owns the inference thread
- **Engine** -- RequestQueue, HistoryManager, ToolRegistry, AgenticLoop, ErrorRecovery
- **Backend** -- abstract `IBackend` interface; production LlamaBackend + MockBackend for testing

See [docs/architecture.md](docs/architecture.md) for the full design.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Prerequisites, build, hello-world agent, core API overview |
| [Tools](docs/tools.md) | Template registration, supported types, manual schema, error recovery |
| [Context Database](docs/context-database.md) | SQLite long-term memory, pruning, FTS5 retrieval |
| [RAG Retrieval](docs/rag.md) | IRetriever interface, InMemoryRagStore, ephemeral injection |
| [Architecture](docs/architecture.md) | Three-layer design, threading model, design principles |
| [Configuration](docs/configuration.md) | Config fields, sampling params, templates, ChatOptions |
| [Examples](docs/examples.md) | Streaming, tools, RAG, context DB, error handling, cancellation |
| [MCP](docs/mcp.md) | Model Context Protocol client, tool federation, transport |
| [Building](docs/building.md) | CMake options, platform setup, sanitizers, coverage |

## Testing

316 unit tests using GoogleTest:

```bash
ctest --test-dir build --output-on-failure
```

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann
- [tl::expected](https://github.com/TartanLlama/expected) by Sy Brand

## License

MIT
