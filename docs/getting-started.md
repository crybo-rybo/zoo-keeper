# Getting Started

This guide walks through the split public API and a minimal first agent.

## Prerequisites

- **C++23 compiler**: macOS uses Clang 16+; Linux uses GCC 13+ or Clang 18+
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
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

See [building.md](building.md) for platform-specific setup, integration tests, and package-install usage.

## Your First Agent

```cpp
#include <zoo/zoo.hpp>
#include <iostream>

int main() {
    zoo::ModelConfig model;
    model.model_path = "models/llama-3-8b.gguf";
    model.context_size = 8192;
    model.n_gpu_layers = 0;

    zoo::AgentConfig agent;
    agent.max_history_messages = 32;

    zoo::GenerationOptions generation;
    generation.max_tokens = 256;

    auto result = zoo::Agent::create(model, agent, generation);
    if (!result) {
        std::cerr << result.error().to_string() << '\n';
        return 1;
    }
    auto agent_runtime = std::move(*result);

    agent_runtime->set_system_prompt("You are a helpful AI assistant.");

    auto handle = agent_runtime->chat(zoo::MessageView{zoo::Role::User, "Hello!"});
    auto response = handle.await_result();
    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
```

## Core API Overview

### `zoo::Agent`

Create the async orchestration layer with `Agent::create(model_config, agent_config, generation)`.

| Method | Description |
|--------|-------------|
| `create(model, agent, generation)` | Validate the split config blocks, load the model, and start the inference thread |
| `chat(message)` | Submit a user message, returns `RequestHandle<TextResponse>` |
| `chat(message, callback)` | Chat with a per-token streaming callback |
| `complete(messages)` | Submit a stateless request-scoped history without mutating retained history |
| `complete(messages, callback)` | Stateless completion with a per-token streaming callback |
| `extract(schema, message)` | Submit a grammar-constrained extraction, returns `RequestHandle<ExtractionResponse>` |
| `extract(schema, messages)` | Stateless extraction with explicit message history |
| `cancel(id)` | Cancel a pending request by ID |
| `set_system_prompt(text)` | Set or update the system prompt |
| `register_tool(name, desc, params, func)` | Register a typed callable as a tool |
| `register_tool(name, desc, schema, handler)` | Register a JSON-backed tool with an explicit schema |
| `stop()` | Gracefully shut down the agent |
| `is_running()` | Check if the agent is accepting requests |
| `clear_history()` | Clear conversation history |
| `get_history()` | Get an owning `HistorySnapshot` of the current conversation |
| `model_config()` | Access the loaded `ModelConfig` |
| `agent_config()` | Access the loaded `AgentConfig` |
| `default_generation_options()` | Access the default `GenerationOptions` |
| `tool_count()` | Number of registered tools |

### `RequestHandle<Result>`

Returned by async agent methods. Use `id()` to correlate or cancel a request, `ready()` to poll, and `await_result()` to block until the `Expected<Result>` is ready.

### `zoo::core::Model`

The synchronous llama.cpp wrapper for direct, single-threaded inference.

| Method | Description |
|--------|-------------|
| `load(model, generation)` | Factory: validate and load the model via the backend |
| `generate(user_message)` | Generate a response and append it to retained history |
| `generate_from_history()` | Generate from the current history state |
| `set_system_prompt(text)` | Set or update the system prompt |
| `add_message(message)` | Add a `MessageView` to history |
| `get_history()` | Get a `HistorySnapshot` copy of the retained conversation |
| `clear_history()` | Clear history and KV cache |
| `context_size()` | Get context window size |
| `estimated_tokens()` | Get estimated token count of history |
| `is_context_exceeded()` | Check if history exceeds the context window |

### `zoo::MessageView`, `ConversationView`, and `HistorySnapshot`

`MessageView` is the borrowed request-scoped message type. `ConversationView` is a borrowed sequence of `MessageView` values used for `complete()` and stateless `extract()` calls. `HistorySnapshot` owns retained history and is what `Model::get_history()` and `Agent::get_history()` return.

Use `HistorySnapshot::view()` when you want to pass retained history back into a request-scoped API without copying the messages again.

### Response Types

`chat()` and `complete()` return `TextResponse`. `extract()` returns `ExtractionResponse`.

- `TextResponse::text` - generated response text
- `TextResponse::usage` - prompt, completion, and total token counts
- `TextResponse::metrics` - latency, time-to-first-token, and throughput
- `TextResponse::tool_trace` - optional tool diagnostics when `GenerationOptions::record_tool_trace` is enabled
- `ExtractionResponse::text` - raw JSON text returned by the model
- `ExtractionResponse::data` - parsed structured output
- `ExtractionResponse::tool_trace` - optional tool diagnostics for extraction calls

## Error Handling

All fallible operations return `Expected<T>` (an alias for `std::expected<T, Error>`). Errors carry a categorized `ErrorCode` and human-readable message:

```cpp
auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "Hello"});
auto response = handle.await_result();
if (!response) {
    std::cerr << response.error().to_string() << '\n';
}
```

## Next Steps

- [Tool System](tools.md) -- register native C++ functions as model-callable tools
- [Configuration Reference](configuration.md) -- all config options
- [Architecture](architecture.md) -- three-layer design and threading model
- [Examples Cookbook](examples.md) -- copy-paste code snippets
