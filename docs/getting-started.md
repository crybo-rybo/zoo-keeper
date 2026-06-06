# Getting Started

This guide walks through building Zoo-Keeper and running a first `zoo::Model`
session.

## Prerequisites

- **C++23 compiler**: macOS uses Clang 16+; Linux uses GCC 13+ or Clang 18+
- **CMake 3.18+**
- **Git**
- **macOS or Linux**
- **Network access on first build** - CMake fetches llama.cpp automatically at configure time.

## Build

```bash
git clone https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

## First Model

Runnable source: [`examples/minimal_model.cpp`](../examples/minimal_model.cpp)

```bash
./build/examples/minimal_model /path/to/model.gguf
```

Minimal API shape:

```cpp
#include <zoo/zoo.hpp>

zoo::ModelConfig model_config;
model_config.model_path = "/path/to/model.gguf";

zoo::GenerationOptions generation;
generation.max_tokens = 128;

auto model = zoo::Model::load(model_config, generation).value();
model->set_system_prompt("You are a helpful assistant.");

auto response = model->generate("Hello!").value();
std::cout << response.text << '\n';
```

## `zoo::Model`

`zoo::Model` is the main llama.cpp-backed harness type. It is synchronous and
owns one model/context/session.

| Method | Description |
|--------|-------------|
| `load(model, generation)` | Validate config, load the GGUF model, create the llama context and sampler |
| `generate(user_message)` | Append a user message, generate an assistant response, and commit it to history |
| `complete(messages)` | Generate against an explicit `ConversationView` without mutating retained history |
| `extract(schema, message)` | Generate schema-constrained JSON and commit the extraction turn |
| `extract(schema, messages)` | Stateless schema-constrained extraction over explicit messages |
| `generate_from_history()` | Generate from retained history and commit the assistant turn, including structured tool calls |
| `set_system_prompt(text)` | Set or replace the leading system prompt |
| `add_message(message)` | Add a structured message to retained history |
| `get_history()` | Return a `HistorySnapshot` copy |
| `clear_history()` | Clear retained history and KV cache |
| `set_tool_calling(specs)` | Configure native llama.cpp template tool-call formatting |
| `parse_tool_response(text)` | Parse native tool-call output into structured calls |
| `context_size()` / `estimated_tokens()` | Inspect model/session token limits and history estimate |

Per-call overrides use `GenerationOverride`. Pass
`GenerationOverride::inherit_defaults()` to use configured defaults, or
`GenerationOverride::explicit_options(options)` to apply an exact
`GenerationOptions` value.

## Streaming And Cancellation

`TokenCallback` streams visible token chunks. `CancellationCallback` lets the
caller stop generation cooperatively.

```cpp
auto on_token = [](std::string_view token) {
    std::cout << token << std::flush;
    return zoo::TokenAction::Continue;
};
auto should_cancel = [&] { return stop_requested.load(); };

auto response = model->generate("Write a short note.", {}, on_token, should_cancel);
```

Returning `TokenAction::Stop` from `on_token` stops after the current streamed
token. Returning `true` from `should_cancel` fails the request with
`ErrorCode::RequestCancelled`.

## Messages And History

`MessageView` is the borrowed request-scoped message type. `ConversationView`
is a borrowed sequence used by `complete()` and stateless `extract()`.
`OwnedMessage` and `HistorySnapshot` own retained conversation state.

Use `HistorySnapshot::view()` to pass retained history back into a request-scoped
API without copying messages again.

## Response Types

`generate()` and `complete()` return `TextResponse`.
`extract()` returns `ExtractionResponse`.

- `TextResponse::text` - generated response text
- `TextResponse::usage` - prompt, completion, and total token counts
- `TextResponse::metrics` - latency, time-to-first-token, and throughput
- `ExtractionResponse::text` - raw JSON text returned by the model
- `ExtractionResponse::data` - parsed structured output

## Error Handling

All fallible operations return `Expected<T>`:

```cpp
auto response = model->generate(zoo::MessageView{zoo::Role::User, "Hello"});
if (!response) {
    std::cerr << response.error().to_string() << '\n';
}
```

## Next Steps

- [Configuration Reference](configuration.md)
- [Structured Output](extract.md)
- [Tool System](tools.md)
- [Examples](../examples/README.md)
