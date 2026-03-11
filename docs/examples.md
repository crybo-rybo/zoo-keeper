# Examples Cookbook

Complete, copy-paste code snippets for common Zoo-Keeper patterns, plus matching executable examples under `examples/`.

## Example Executables

Build all example binaries with:

```bash
cmake -B build -DZOO_BUILD_EXAMPLES=ON
cmake --build build
```

Available programs:

- `demo_chat` -- interactive CLI chat loop driven by a JSON config file
- `model_generate` -- synchronous `zoo::core::Model` usage
- `error_handling` -- structured runtime error reporting
- `stream_cancel` -- streaming output with cooperative cancellation
- `manual_tool_schema` -- `Agent::register_tool(..., schema, handler)` with the supported schema subset

## JSON Config Files

The `demo_chat` executable loads `zoo::Config` from JSON through `zoo/core/json.hpp` and adds one example-only field:

- `tools` -- enables or disables the bundled demo tools

Use [`examples/config.example.json`](../examples/config.example.json) as the starting point. The library itself does not expand `~` or perform file-path normalization; `model_path` is read literally.

## Streaming Output

Print tokens as they arrive:

```cpp
auto handle = agent->chat(
    zoo::Message::user("Write a haiku about AI"),
    [](std::string_view token) {
        std::cout << token << std::flush;
    }
);

auto response = handle.future.get();
std::cout << std::endl;
```

## Multi-Turn Conversation

History is managed automatically:

```cpp
agent->chat(zoo::Message::user("My name is Alice")).future.get();
auto handle = agent->chat(zoo::Message::user("What's my name?"));
auto response = handle.future.get();
// response->text will reference "Alice"
```

## Tool Registration and Calling

```cpp
// Define tools
int add(int a, int b) { return a + b; }
std::string get_time() {
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

// Register
agent->register_tool("add", "Add two integers", {"a", "b"}, add);
agent->register_tool("get_time", "Get current date and time", {}, get_time);

// The model can now call these tools
auto handle = agent->chat(zoo::Message::user("What is 42 + 58?"));
auto response = handle.future.get();
if (response) {
    std::cout << response->text << std::endl;

    // Inspect explicit tool invocation records captured during the agentic loop
    for (const auto& invocation : response->tool_invocations) {
        std::cout << "Tool: " << invocation.name << std::endl;
        std::cout << "Args: " << invocation.arguments_json << std::endl;

        if (invocation.result_json) {
            std::cout << "Result: " << *invocation.result_json << std::endl;
        }

        if (invocation.error) {
            std::cout << "Error: " << invocation.error->to_string() << std::endl;
        }
    }
}
```

## Lambda Tools

```cpp
agent->register_tool("uppercase", "Convert text to uppercase", {"text"},
    [](std::string text) -> std::string {
        std::transform(text.begin(), text.end(), text.begin(), ::toupper);
        return text;
    });
```

## Error Handling

```cpp
auto handle = agent->chat(zoo::Message::user("Hello"));
auto result = handle.future.get();

if (!result) {
    zoo::Error error = result.error();
    switch (error.code) {
        case zoo::ErrorCode::ContextWindowExceeded:
            std::cerr << "Context full!" << std::endl;
            agent->clear_history();
            break;
        case zoo::ErrorCode::InferenceFailed:
            std::cerr << "Inference failed: " << error.message << std::endl;
            break;
        case zoo::ErrorCode::ToolRetriesExhausted:
            std::cerr << "Tool retries exhausted: " << error.message << std::endl;
            break;
        default:
            std::cerr << error.to_string() << std::endl;
    }
} else {
    std::cout << result->text << std::endl;
}
```

## Cancellation

```cpp
auto handle = agent->chat(zoo::Message::user("Write a long essay"));

// Cancel by request ID
agent->cancel(handle.id);

auto result = handle.future.get();
if (!result && result.error().code == zoo::ErrorCode::RequestCancelled) {
    std::cout << "Generation cancelled" << std::endl;
}
```

The `stream_cancel` executable demonstrates the same pattern end to end with a real `Agent`.

## Metrics

```cpp
auto handle = agent->chat(zoo::Message::user("Hello"));
auto response = handle.future.get();
if (response) {
    std::cout << "Latency: " << response->metrics.latency_ms.count() << " ms" << std::endl;
    std::cout << "TTFT: " << response->metrics.time_to_first_token_ms.count() << " ms" << std::endl;
    std::cout << "Speed: " << response->metrics.tokens_per_second << " tok/s" << std::endl;
    std::cout << "Tokens: " << response->usage.prompt_tokens << " prompt + "
              << response->usage.completion_tokens << " completion" << std::endl;
}
```

## Using Model Directly (Layer 1)

For synchronous, single-threaded usage without the Agent layer:

```cpp
#include <zoo/core/model.hpp>
#include <iostream>

int main() {
    zoo::Config config;
    config.model_path = "models/llama-3-8b.gguf";
    config.context_size = 8192;
    config.max_tokens = 256;

    auto result = zoo::core::Model::load(config);
    if (!result) {
        std::cerr << result.error().to_string() << std::endl;
        return 1;
    }
    auto& model = *result;

    model->set_system_prompt("You are a helpful assistant.");

    auto response = model->generate("What is the capital of France?");
    if (response) {
        std::cout << response->text << std::endl;
        std::cout << "Tokens: " << response->usage.total_tokens << std::endl;
    }
}
```

The `model_generate` executable is the standalone version of this pattern.

## See Also

- [Getting Started](getting-started.md) -- setup walkthrough
- [Tools](tools.md) -- tool system deep-dive
- [Configuration](configuration.md) -- all config options
