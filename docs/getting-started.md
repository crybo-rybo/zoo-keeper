# Getting Started

This guide walks you through setting up Zoo-Keeper and building your first AI agent.

## Prerequisites

- **C++17 compiler**: GCC 11+, Clang 13+, or MSVC 2019+
- **CMake 3.18+**
- **Git** (for submodules)

## Installation

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper

# If you already cloned without submodules:
git submodule update --init --recursive
```

## Building

```bash
cmake -B build -DZOO_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

See [building.md](building.md) for platform-specific setup (Metal, CUDA) and advanced options.

## Your First Agent

```cpp
#include <zoo/zoo.hpp>
#include <iostream>

int main() {
    // 1. Configure
    zoo::Config config;
    config.model_path = "models/llama-3-8b.gguf";
    config.context_size = 8192;
    config.max_tokens = 512;
    config.prompt_template = zoo::PromptTemplate::Llama3;

    // 2. Create the agent
    auto result = zoo::Agent::create(config);
    if (!result) {
        std::cerr << result.error().to_string() << std::endl;
        return 1;
    }
    auto agent = std::move(*result);

    // 3. Set a system prompt
    agent->set_system_prompt("You are a helpful AI assistant.");

    // 4. Chat — chat() returns a RequestHandle; call .future.get() to block
    auto handle = agent->chat(zoo::Message::user("Hello!"));
    auto response = handle.future.get();

    if (response) {
        std::cout << response->text << std::endl;
    } else {
        std::cerr << response.error().to_string() << std::endl;
    }

    return 0;
}
```

## Core API Overview

### `zoo::Agent`

The single entry point for all library functionality. Created via the `Agent::create()` factory method, which returns `Expected<std::unique_ptr<Agent>>`.

| Method | Description |
|--------|-------------|
| `create(config)` | Factory: validate config, load model, start inference thread |
| `chat(message)` | Submit a message, returns `RequestHandle` (holds `.id` and `.future`) |
| `chat(message, callback)` | Chat with per-token streaming callback |
| `chat(message, options, callback)` | Chat with per-request options (RAG, etc.) |
| `set_system_prompt(text)` | Set or update the system prompt |
| `register_tool(name, desc, params, func)` | Register a callable as a tool |
| `set_retriever(retriever)` | Install a RAG retriever |
| `enable_context_database(path)` | Enable SQLite long-term memory |
| `add_mcp_server(config)` | Connect to an MCP server and federate its tools |
| `remove_mcp_server(server_id)` | Disconnect and remove one MCP server |
| `list_mcp_servers()` | List MCP server summaries (id, connected, tool count) |
| `get_mcp_server(server_id)` | Retrieve one MCP server summary by ID |
| `mcp_server_count()` | Number of connected MCP servers |
| `stop()` | Gracefully shut down the agent |
| `clear_history()` | Clear conversation history |
| `get_history()` | Get a copy of current conversation messages |

### `zoo::Message`

Value type representing a conversation turn. Use the factory methods:

```cpp
Message::system("You are helpful.")
Message::user("What is 2+2?")
Message::assistant("4")
Message::tool("result", "call_id")
```

### `zoo::Response`

Returned from `chat()` via `std::future`. Contains:

- `text` -- generated response text
- `usage` -- token counts (prompt, completion, total)
- `metrics` -- latency, time-to-first-token, tokens/sec
- `tool_calls` -- tool call and result history
- `rag_chunks` -- retrieved RAG chunks used for this turn

### Error Handling

All fallible operations return `Expected<T>` (an alias for `tl::expected<T, Error>`). Errors carry a categorized `ErrorCode` and human-readable message:

```cpp
auto response = agent->chat(zoo::Message::user("Hello")).future.get();
if (!response) {
    std::cerr << response.error().to_string() << std::endl;
}
```

## Next Steps

- [Tool System](tools.md) -- register native C++ functions as model-callable tools
- [RAG Retrieval](rag.md) -- inject external knowledge into prompts
- [Context Database](context-database.md) -- long-term memory for extended conversations
- [Configuration Reference](configuration.md) -- all config options
- [Examples Cookbook](examples.md) -- copy-paste code snippets
