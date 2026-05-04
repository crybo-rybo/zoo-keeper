<p align="center">
  <img src="docs/images/zoo_keeper_logo.png" alt="Zoo-Keeper logo" width="220">
</p>

<h1 align="center">Zoo-Keeper</h1>

<p align="center">
  <b>The C++23 SDK for embedding local LLMs into your applications.</b><br/>
  <sub>Async agent runtime &bull; Native tool calling &bull; Structured output &bull; Zero network dependency</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-23-blue" alt="C++23" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/tests-ctest%20passing-success" alt="Tests" />
</p>

---

## What is Zoo-Keeper?

Zoo-Keeper is a C++23 inference SDK built on [llama.cpp](https://github.com/ggerganov/llama.cpp). It turns the raw llama.cpp C API into a layered, type-safe library designed to be embedded into applications — desktop software, game engines, developer tools, embedded systems, or anything that needs local LLM inference without a server.

Think of it this way: **llama.cpp is the engine. Zoo-Keeper is the SDK.**

```cpp
// Five lines from zero to a running agent with tools
auto agent = zoo::Agent::create(config).value();
agent->set_system_prompt("You are a helpful assistant.");
agent->register_tool("search", "Search the web", {"query"}, my_search_fn);
auto handle = agent->chat("Find flights to Tokyo", {}, on_token);
auto result = handle.await_result().value();
```

## Why Zoo-Keeper over raw llama.cpp?

llama.cpp is an exceptional inference engine, but it exposes a flat C API with ~200+ functions, manual resource management, and no application-level abstractions. Building a real application on it means writing hundreds of lines of threading, state management, and error handling before you get to your first inference call.

Zoo-Keeper closes that gap.

### Side-by-side: adding a tool-calling chat agent

<table>
<tr><th>Raw llama.cpp</th><th>Zoo-Keeper</th></tr>
<tr>
<td>

- Manually initialize backend, load model, create context, build sampler chain
- Implement chat history as a vector of `llama_chat_message` structs
- Render prompts via `llama_chat_apply_template()`, track incremental offsets
- Tokenize, batch, decode in a manual loop with `llama_batch` and `llama_decode`
- Parse tool calls from raw text output (format-specific)
- Execute tool handlers, format results, re-render, re-decode
- Manage KV cache state across turns
- Build your own threading model for async inference
- Handle every error as a null pointer or negative int

</td>
<td>

```cpp
auto agent = zoo::Agent::create(config).value();
agent->register_tool("add", "Add numbers",
    {"a", "b"}, [](int a, int b) { return a + b; });
auto handle = agent->chat("What is 2+2?");
auto response = handle.await_result().value();
// Tool was called, result fed back, final
// answer generated — all automatically.
```

</td>
</tr>
</table>

### What you get

| Capability | Raw llama.cpp | Zoo-Keeper |
|-----------|:---:|:---:|
| Model loading and inference | Manual C API | `Model::load()` with validated config |
| Async inference with streaming | Build your own | `Agent::chat()` returns `RequestHandle<T>` |
| Request cancellation | Implement yourself | `handle.cancel()` or `agent->cancel(handle.id())` |
| Chat history with KV cache sync | Manual bookkeeping | Built into `Model`, auto-trimmed |
| Tool calling (llama.cpp PEG formats) | Parse text yourself | Template-driven detection + execution loop |
| Type-safe tool registration | N/A | `register_tool("name", desc, params, callable)` |
| Tool argument validation | N/A | Automatic JSON Schema validation with retries |
| Structured output extraction | Build grammar yourself | `agent->extract(schema, prompt)` |
| Error handling | Null pointers, `-1` returns | `std::expected<T, Error>` with categorized codes |
| Thread safety | Your problem | Agent owns inference thread; callers submit requests |
| Streaming callbacks | Wire up yourself | Token callbacks on `chat()`, `complete()`, `extract()` |
| Response metrics | Compute yourself | `TextResponse` includes latency, throughput, token usage |

### What you don't get (by design)

Zoo-Keeper compiles llama.cpp with `LLAMA_BUILD_TOOLS=OFF` and `LLAMA_BUILD_EXAMPLES=OFF`. Your application links only the inference core and utility libraries — no llama-server, no llama-cli, no quantization tools. The result is a focused static library (~30-50 MB) instead of the full llama.cpp distribution (~100-150 MB).

## Architecture

Four layers with strict downward-only dependencies. Use only what you need:

Each layer depends only on the layers below it. Consumers can stop at whichever level fits their needs.

| Layer | Namespace | What it does | Key types |
|-------|-----------|-------------|-----------|
| **Hub** *(optional)* | `zoo::hub` | GGUF inspection, HuggingFace downloads, local model store, auto-configuration | `GgufInspector`, `ModelStore`, `HuggingFaceClient` |
| **Agent** | `zoo::Agent` | Async inference runtime with request queue, per-token streaming, cancellation, agentic tool loop, and structured extraction | `Agent`, `RequestHandle<T>`, `TextResponse`, `ExtractionResponse` |
| **Tools** | `zoo::tools` | Tool registration, JSON Schema generation from C++ signatures, argument validation. Zero llama.cpp dependency | `ToolRegistry`, `ToolCallParser`, `ToolArgumentsValidator` |
| **Core** | `zoo::core` | Direct synchronous llama.cpp wrapper — model loading, prompt rendering, generation, chat history, KV cache management | `Model`, `ModelConfig`, `GenerationOptions` |

**Threading model:** The Agent owns a single inference thread. Callers submit requests via `chat()`, `complete()`, or `extract()` and receive a `RequestHandle<T>`. Model access is confined to that thread, streaming callbacks run on a callback dispatcher, and tool handlers run on a tool executor worker.

## Use Cases

**Desktop applications** — Ship local AI features without requiring users to install Python, run a server, or sign up for an API. Zoo-Keeper links as a static library alongside your application.

**Developer tools** — Build code assistants, local copilots, or CLI tools with on-device inference and tool calling. The Agent runtime handles the full request-tool-response loop.

**Game engines and creative tools** — Integrate NPCs, procedural content generation, or in-context assistance with predictable latency and streaming token delivery.

**Edge and embedded systems** — Run quantized models on constrained hardware. Zoo-Keeper's CPU-first defaults and optional GPU offloading (Metal, CUDA) adapt to the target platform.

**Prototyping and research** — Iterate on agentic workflows with native tools, structured extraction, and full control over sampling parameters — without the overhead of a REST API layer.

## Quick Start

### Build

```bash
git clone https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

llama.cpp is fetched automatically at CMake configure time — no submodules
or extra setup required.

### Integrate via CMake

```cmake
include(FetchContent)
FetchContent_Declare(zoo-keeper
    GIT_REPOSITORY https://github.com/crybo-rybo/zoo-keeper.git
    GIT_TAG        v1.1.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(zoo-keeper)

target_link_libraries(my_app PRIVATE ZooKeeper::zoo)
```

If your parent project already defines both `llama` and `llama-common` CMake
targets, Zoo-Keeper reuses them automatically and skips its own fetch.

### Run your first agent

```cpp
#include <iostream>
#include <zoo/zoo.hpp>

int main() {
    zoo::ModelConfig model;
    model.model_path = "models/llama-3-8b.gguf";
    model.context_size = 8192;
    model.n_gpu_layers = -1; // Offload all layers to GPU

    auto agent = zoo::Agent::create(model).value();
    if (auto set_prompt = agent->try_set_system_prompt("You are a concise assistant."); !set_prompt) {
        std::cerr << set_prompt.error().to_string() << '\n';
        return 1;
    }

    // Register a native C++ function as a tool
    agent->register_tool("add", "Add two integers", {"a", "b"},
        [](int a, int b) { return a + b; });

    // Stream tokens as they arrive
    auto on_token = [](std::string_view token) {
        std::cout << token << std::flush;
        return zoo::TokenAction::Continue;
    };

    auto handle =
        agent->chat("What is 42 + 58?", zoo::GenerationOverride::inherit_defaults(), on_token);
    auto response = handle.await_result();

    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << "\n\nTokens: " << response->usage.total_tokens
              << " | " << response->metrics.tokens_per_second << " tok/s\n";
}
```

## Feature Highlights

### Native tool calling

Register any C++ callable and Zoo-Keeper generates the JSON Schema, detects tool calls from llama.cpp PEG parser output, validates arguments, executes the handler, and feeds results back into the conversation:

```cpp
agent->register_tool("get_weather", "Get current weather", {"city"},
    [](std::string city) -> std::string {
        return fetch_weather(city);  // Your code
    });

// The agent automatically:
// 1. Detects the model wants to call get_weather
// 2. Validates {"city": "Tokyo"} against the schema
// 3. Calls your lambda
// 4. Feeds the result back to the model
// 5. Generates the final response
```

### Structured output extraction

Constrain model output to a JSON Schema using grammar-guided generation:

```cpp
nlohmann::json schema = nlohmann::json::parse(
    R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}})");
auto handle = agent->extract(schema, "Extract info: John is 30 years old");
auto result = handle.await_result().value();

// result.data == {"name": "John", "age": 30}
```

### Model hub (optional)

Build with `ZOO_BUILD_HUB=ON` for GGUF inspection, HuggingFace downloading, and a local model store. Downloads share the llama.cpp cache — models fetched by any llama.cpp tool are automatically available:

```cpp
auto store = zoo::hub::ModelStore::open().value();
auto hf = zoo::hub::HuggingFaceClient::create().value();

// Pull a model (ETag caching, resume, split-file support)
store->pull(*hf, "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", {"llama3"});

// One-liner from alias to running agent
auto agent = store->create_agent("llama3").value();
```

### Error handling

Every fallible operation returns `Expected<T>` (`std::expected<T, Error>`). No exceptions, no null checks, no mystery crashes:

```cpp
auto result = zoo::Agent::create(config);
if (!result) {
    // Categorized error codes: InvalidModelPath, ModelLoadFailed,
    // ContextCreationFailed, ToolValidationFailed, ...
    std::cerr << result.error().to_string() << '\n';
}
```

## API Surface

| Type | Purpose |
|------|---------|
| `zoo::core::Model` | Direct synchronous llama.cpp wrapper — model loading, generation, history, KV cache |
| `zoo::Agent` | Async runtime — request queue, streaming, tool loop, structured extraction |
| `zoo::tools::ToolRegistry` | Deterministic tool registration with typed callables or JSON Schema handlers; externally synchronize direct multi-threaded use |
| `zoo::RequestHandle<T>` | Async result handle with `id()`, `ready()`, `await_result()`, cancellation |
| `zoo::TextResponse` | Generated text + token usage + latency metrics + optional tool trace |
| `zoo::ExtractionResponse` | Parsed JSON output + raw text + usage + metrics |
| `zoo::ModelConfig` / `zoo::AgentConfig` / `zoo::GenerationOptions` | Validated configuration with JSON serialization |
| `zoo::hub::GgufInspector` | GGUF metadata reading without loading weights |
| `zoo::hub::ModelStore` | Local model catalog with aliases and auto-configuration |
| `zoo::hub::HuggingFaceClient` | HuggingFace downloading with shared llama.cpp cache |

## Testing

```bash
scripts/test.sh                     # Unit tests (pure logic, no model needed)

# Hub-layer unit tests are only built when the hub is enabled
scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON
scripts/test.sh -R "HuggingFace|ModelStore|AutoConfig|GgufInspector|HubPath"

# Integration tests (requires a real GGUF model)
scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON
ZOO_INTEGRATION_MODEL=/path/to/model.gguf scripts/test.sh
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | First build, first agent, core API walkthrough |
| [Building](docs/building.md) | CMake setup, FetchContent, Metal/CUDA, sanitizers, install/package |
| [Configuration](docs/configuration.md) | Model config, sampling parameters, generation limits, history budgets |
| [Tools](docs/tools.md) | Typed tools, manual schema registration, supported schema subset, error handling |
| [Structured Output](docs/extract.md) | Grammar-constrained extraction, schema reference, stateful vs. stateless |
| [Hub Layer](docs/hub.md) | GGUF inspection, HuggingFace downloading, local model store, auto-configuration |
| [Architecture](docs/architecture.md) | Layer design, runtime ownership, threading model, target structure |
| [Examples](docs/examples.md) | Streaming, cancellation, tools, error handling, model store |
| [Compatibility](docs/compatibility.md) | Public API boundary, 1.x stability policy, deprecation rules |
| [Migration](MIGRATION.md) | Upgrade notes for major API changes |

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov — the inference engine beneath the SDK
- [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann
- [GoogleTest](https://github.com/google/googletest) by Google

## License

MIT
