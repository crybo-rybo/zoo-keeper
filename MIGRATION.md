# Migration Guide

This document covers what consumers need to know when upgrading Zoo-Keeper.

## v1.0.3 → v1.1.0

Version `1.1.0` is a major API step rather than a small patch. The biggest
changes are the config split, typed request handles, dedicated extraction
results, and the clearer message/history APIs that separate retained state from
request-scoped conversations.

### Config Split

Before, consumers configured the library through one aggregate `zoo::Config`
object. After this release, the configuration concerns are split into
`ModelConfig`, `AgentConfig`, and `GenerationOptions`.

```cpp
// Before: one aggregate config object.
zoo::Config config;
config.model_path = "/models/llama.gguf";
config.context_size = 8192;
config.max_history_messages = 64;
config.max_tokens = 128;

auto agent = zoo::Agent::create(config);
```

```cpp
// After: pass the three configuration concerns explicitly.
zoo::ModelConfig model_config;
model_config.model_path = "/models/llama.gguf";
model_config.context_size = 8192;

zoo::AgentConfig agent_config;
agent_config.max_history_messages = 64;

zoo::GenerationOptions generation;
generation.max_tokens = 128;

auto agent = zoo::Agent::create(model_config, agent_config, generation);
```

### Request Handles

Async calls still return request handles, but the handle surface changed.
Before, the handle exposed `.id` and a `std::future`; now it exposes `id()`,
`ready()`, and `await_result()`.

```cpp
// Before: request completion came from the embedded future.
auto handle = agent->chat(zoo::Message::user("Hello"));
auto response = handle.future.get();
if (response) {
    std::cout << response->text << "\n";
}
```

```cpp
// After: the call returns a request handle.
auto handle = agent->chat("Hello");
auto response = handle.await_result();
if (response) {
    std::cout << response->text << "\n";
}
```

### Extraction Results

Structured extraction now returns `ExtractionResponse`, which separates the raw model
text from the parsed JSON payload.

```cpp
// Before: extracted data lived on the generic response type.
auto handle = agent->extract(schema, zoo::Message::user("Alice is 30."));
auto response = handle.future.get();
if (response) {
    std::cout << response->extracted_data->dump() << "\n";
}
```

```cpp
// After: extraction data is part of the dedicated extraction response type.
auto handle = agent->extract(schema, "Alice is 30.");
auto response = handle.await_result();
if (response) {
    std::cout << response->data.dump() << "\n";
    std::cout << response->text << "\n";
}
```

### Message And History APIs

Request-scoped inputs now use `MessageView` and `ConversationView`, while
retained agent state stays behind `get_history()` and `clear_history()`. Use
`chat()` for appending a new turn and `complete()` for running against a
supplied history without mutating the retained conversation.

```cpp
// Before: scoped history passed owning Message values directly.
std::vector<zoo::Message> history = {
    zoo::Message::user("Hello"),
    zoo::Message::assistant("Hi there"),
    zoo::Message::user("What did I just say?")
};

auto scoped = agent->complete(history);
```

```cpp
// After: choose the request shape explicitly.
agent->chat(zoo::MessageView{zoo::Role::User, "Hello"});

const std::array<zoo::MessageView, 3> history = {
    zoo::MessageView{zoo::Role::User, "Hello"},
    zoo::MessageView{zoo::Role::Assistant, "Hi there"},
    zoo::MessageView{zoo::Role::User, "What did I just say?"},
};

auto scoped = agent->complete(
    zoo::ConversationView{std::span<const zoo::MessageView>(history)});
```

### Tool Calling

Tool calling is now template-driven through llama.cpp's native chat templates. Models
that do not expose a native tool format no longer rely on the old sentinel-based
fallback path.

## 0.2.x → 1.0.0

### Summary

No breaking API changes occurred between 0.2.x and 1.0.0. The public interface was stable throughout the pre-1.0 development cycle. A source-compatible upgrade is expected for any consumer that stayed within the documented public boundary.

### Public Boundary

The supported public API is:

- Headers under `include/zoo/` (excluding `include/zoo/internal/`)
- The primary CMake target `ZooKeeper::zoo`

Internal headers (`include/zoo/internal/`), source files (`src/`), and CMake packaging internals are not part of the compatibility contract and may change in any release.

### CMake Target Changes

`ZooKeeper::zoo` is and has been the primary consumer target throughout 0.2.x and into 1.0.0.

`ZooKeeper::zoo_core` remains available as a compatibility alias that forwards to `ZooKeeper::zoo`. New consumers should use `ZooKeeper::zoo` directly.

### C++ Standard

C++23 is required. This has not changed from 0.2.x.

### llama.cpp Submodule

The `extern/llama.cpp` submodule is now pinned to a specific commit (`d1b4757dedbb60a811c8d7012249a96b1b702606`, tagged `gguf-v0.17.1-2054-gd1b4757de`) for reproducible builds. Consumers embedding Zoo-Keeper via `add_subdirectory` or `FetchContent` will get this pinned version automatically.

### What Has Not Changed

- All public headers: `zoo/zoo.hpp`, `zoo/agent.hpp`, `zoo/core/model.hpp`, `zoo/core/types.hpp`, `zoo/tools/registry.hpp`, `zoo/tools/parser.hpp`, `zoo/tools/validation.hpp`
- All public types: `zoo::Agent`, `zoo::core::Model`, `zoo::Message`, `zoo::Role`, `zoo::Config`, `zoo::Response`, `zoo::Error`
- Error handling: `std::expected`-based throughout
- Include paths: unchanged

## Future Releases

For post-1.0 migration notes, see [CHANGELOG.md](CHANGELOG.md) and subsequent entries in this file.
