# Examples Cookbook

Copy-paste snippets for common Zoo-Keeper patterns, plus the matching executables under `examples/`.

## Example Executables

Build the example binaries with:

```bash
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

Available programs:

- `demo_chat` - interactive CLI chat loop driven by a nested JSON config file
- `demo_extract` - structured extraction examples for stateful, stateless, and streaming flows
- `model_generate` - synchronous `zoo::core::Model` usage
- `error_handling` - structured runtime error reporting
- `stream_cancel` - streaming output with cooperative cancellation
- `manual_tool_schema` - `Agent::register_tool(..., schema, handler)` with the supported schema subset

## JSON Config Files

`demo_chat` loads a top-level JSON wrapper with nested `model`, `agent`, and `generation` blocks, then applies `system_prompt` and the example-only `tools` toggle. See [`examples/config.example.json`](../examples/config.example.json) for the exact shape.

The library itself serializes `zoo::ModelConfig`, `zoo::AgentConfig`, and `zoo::GenerationOptions` directly through `zoo/core/json.hpp`.

## Streaming Output

Print tokens as they arrive:

```cpp
auto handle = agent->chat(
    zoo::MessageView{zoo::Role::User, "Write a haiku about AI"},
    {},
    [](std::string_view token) {
        std::cout << token << std::flush;
    }
);

auto response = handle.await_result();
std::cout << '\n';
```

## Multi-Turn Conversation

Retained history is managed automatically:

```cpp
agent->chat(zoo::MessageView{zoo::Role::User, "My name is Alice"}).await_result();
auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "What's my name?"});
auto response = handle.await_result();
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

// The model can now call these tools. Enable record_tool_trace when you want
// the runtime to retain the tool-attempt diagnostics.
zoo::GenerationOptions options;
options.record_tool_trace = true;

auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "What is 42 + 58?"}, options);
auto response = handle.await_result();
if (response) {
    std::cout << response->text << std::endl;

    if (response->tool_trace) {
        for (const auto& invocation : response->tool_trace->invocations) {
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
auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "Hello"});
auto result = handle.await_result();

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
auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "Write a long essay"});

// Cancel by request ID
agent->cancel(handle.id());

auto result = handle.await_result();
if (!result && result.error().code == zoo::ErrorCode::RequestCancelled) {
    std::cout << "Generation cancelled" << std::endl;
}
```

The `stream_cancel` executable demonstrates the same pattern end to end with a real `Agent`.

## Structured Extraction

`demo_extract` demonstrates the same extraction flow in a runnable program. The
minimal pattern is:

```cpp
nlohmann::json schema = {
    {"type", "object"},
    {"properties", {{"count", {{"type", "integer"}}}}},
    {"required", {"count"}},
    {"additionalProperties", false}
};

auto handle = agent->extract(schema, "There are 7 apples on the shelf.");
auto response = handle.await_result();
if (response) {
    std::cout << response->data["count"] << std::endl;
}
```

## Metrics

```cpp
auto handle = agent->chat(zoo::MessageView{zoo::Role::User, "Hello"});
auto response = handle.await_result();
if (response) {
    std::cout << "Latency: " << response->metrics.latency_ms.count() << " ms" << std::endl;
    std::cout << "TTFT: " << response->metrics.time_to_first_token_ms.count() << " ms" << std::endl;
    std::cout << "Speed: " << response->metrics.tokens_per_second << " tok/s" << std::endl;
    std::cout << "Tokens: " << response->usage.prompt_tokens << " prompt + "
              << response->usage.completion_tokens << " completion" << std::endl;
}
```

## Using Model Directly

For synchronous, single-threaded usage without the agent layer:

```cpp
#include <zoo/core/model.hpp>
#include <iostream>

int main() {
    zoo::ModelConfig model;
    model.model_path = "models/llama-3-8b.gguf";
    model.context_size = 8192;

    zoo::GenerationOptions generation;
    generation.max_tokens = 256;

    auto result = zoo::core::Model::load(model, generation);
    if (!result) {
        std::cerr << result.error().to_string() << std::endl;
        return 1;
    }
    auto& model_runtime = *result;

    model_runtime->set_system_prompt("You are a helpful assistant.");

    auto response = model_runtime->generate("What is the capital of France?");
    if (response) {
        std::cout << response->text << std::endl;
        std::cout << "Tokens: " << response->usage.total_tokens << std::endl;
    }
}
```

The `model_generate` executable uses the same shape.

## See Also

- [Getting Started](getting-started.md) -- setup walkthrough
- [Tools](tools.md) -- tool system deep-dive
- [Configuration](configuration.md) -- all config options
