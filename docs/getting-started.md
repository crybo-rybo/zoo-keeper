# Getting Started

This guide walks you through setting up Zoo-Keeper and building your first AI agent.

## Prerequisites

- **C++23 compiler**: GCC 13+, Clang 16+
- **CMake 3.18+**
- **Git** (for submodules)
- **macOS or Linux**

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

See [building.md](building.md) for platform-specific setup (Metal, CUDA), integration tests, and package-install usage.

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
    config.n_gpu_layers = 0; // opt in to GPU offload explicitly

    // 2. Create the agent
    auto result = zoo::Agent::create(config);
    if (!result) {
        std::cerr << result.error().to_string() << std::endl;
        return 1;
    }
    auto agent = std::move(*result);

    // 3. Set a system prompt
    agent->set_system_prompt("You are a helpful AI assistant.");

    // 4. Chat -- chat() returns a RequestHandle; call .future.get() to block
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

The primary entry point for agentic behavior. Created via the `Agent::create()` factory method, which returns `Expected<std::unique_ptr<Agent>>`.

| Method | Description |
|--------|-------------|
| `create(config)` | Factory: validate config, load model, start inference thread |
| `chat(message)` | Submit a message, returns `RequestHandle` (holds `.id` and `.future`) |
| `chat(message, callback)` | Chat with per-token streaming callback |
| `cancel(id)` | Cancel a pending request by ID |
| `set_system_prompt(text)` | Set or update the system prompt |
| `register_tool(name, desc, params, func)` | Register a typed callable as a tool |
| `register_tool(name, desc, schema, handler)` | Register a JSON-backed tool with an explicit schema |
| `stop()` | Gracefully shut down the agent |
| `is_running()` | Check if the agent is accepting requests |
| `clear_history()` | Clear conversation history |
| `get_history()` | Get a copy of current conversation messages |
| `tool_count()` | Number of registered tools |
| `get_config()` | Returns a const reference to the active configuration |

### `zoo::core::Model`

The synchronous llama.cpp wrapper (Layer 1). Can be used standalone without Agent for direct, single-threaded inference.

| Method | Description |
|--------|-------------|
| `load(config)` | Factory: validate config, load model via backend |
| `generate(user_message)` | Generate a response (adds to history automatically) |
| `generate_from_history()` | Generate from current history state |
| `set_system_prompt(text)` | Set or update the system prompt |
| `add_message(message)` | Add a message to history |
| `get_history()` | Get conversation history |
| `clear_history()` | Clear history and KV cache |
| `context_size()` | Get context window size |
| `estimated_tokens()` | Get estimated token count of history |
| `is_context_exceeded()` | Check if history exceeds context window |

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
- `tool_invocations` -- explicit tool attempts including name, arguments, result, and error outcome

### Error Handling

All fallible operations return `Expected<T>` (an alias for `std::expected<T, Error>`). Errors carry a categorized `ErrorCode` and human-readable message:

```cpp
auto handle = agent->chat(zoo::Message::user("Hello"));
auto response = handle.future.get();
if (!response) {
    std::cerr << response.error().to_string() << std::endl;
}
```

## Next Steps

- [Tool System](tools.md) -- register native C++ functions as model-callable tools
- [Configuration Reference](configuration.md) -- all config options
- [Architecture](architecture.md) -- three-layer design and threading model
- [Examples Cookbook](examples.md) -- copy-paste code snippets
