# Migration Guide

This document covers the v2 hard break that turns Zoo-Keeper into a
llama.cpp-specific LLM harness centered on `zoo::Model`.

## Unreleased v2

Zoo-Keeper no longer presents an Agent SDK as the primary API. The installed
surface is now a synchronous model/session harness with retained history,
stateless completion, structured extraction, streaming callbacks, cancellation
callbacks, token accounting, and native tool-call parsing.

### Include And Construction

- Use `zoo::Model::load(...)` instead of `zoo::Agent::create(...)`.
- Include `zoo/model.hpp` or `zoo/zoo.hpp`; `zoo/agent.hpp` is removed.
- Link `ZooKeeper::zoo`; the compatibility `ZooKeeper::zoo_core` target is
  removed.
- `zoo::core::Model` still exists as the implementation type, but normal
  consumers should spell the type as `zoo::Model`.

```cpp
zoo::ModelConfig config;
config.model_path = "/models/model.gguf";

auto model = zoo::Model::load(config);
if (!model) {
    std::cerr << model.error().to_string() << '\n';
}
```

### Removed Agent SDK Surface

The following public APIs are removed:

- `zoo::Agent`
- `AgentConfig`
- `RequestHandle`
- async request queues
- automatic tool-loop APIs
- `GenerationOptions::record_tool_trace`
- response `tool_trace` fields
- `zoo::AsyncTextCallback`
- `zoo::AsyncTokenCallback`
- `zoo::hub::ModelStore::create_agent(...)`

Use synchronous `Model` calls and application-owned scheduling where the old
Agent runtime previously provided async orchestration.

### Generation

Retained conversation generation is owned by `Model`:

```cpp
model->set_system_prompt("You are concise.");
auto response = model->generate("Hello");
```

Stateless completion is now request-scoped and restores retained history after
the call:

```cpp
std::array messages = {
    zoo::MessageView{zoo::Role::System, "Answer briefly."},
    zoo::MessageView{zoo::Role::User, "What is llama.cpp?"},
};

auto response = model->complete(zoo::ConversationView{std::span(messages)});
```

### Structured Extraction

Structured output moved from `Agent::extract(...)` to synchronous
`Model::extract(...)`:

```cpp
nlohmann::json schema = {
    {"type", "object"},
    {"properties", {
        {"name", {{"type", "string"}}},
        {"age", {{"type", "integer"}}}
    }},
    {"required", {"name", "age"}},
    {"additionalProperties", false}
};

auto extracted = model->extract(schema, "Alice is 30.");
```

### Tools

Zoo-Keeper now treats tools as model-output schemas plus parser/validator
utilities. It does not own executable tools.

- Use `zoo::ToolSpec` for model-facing tool definitions.
- Use `zoo::tools::ToolRegistry` to canonicalize schemas and keep validation
  metadata.
- Use `Model::set_tool_calling(registry.get_all_tool_specs())` to enable native
  template-driven tool calls.
- Use `GenerationResult::tool_calls`, `Model::parse_tool_response(...)`, and
  `ToolArgumentsValidator` to parse and validate model output.
- Dispatch validated tool calls in your application code, then add
  `OwnedMessage::tool(...)` if you want another model pass.

Removed tool execution helpers:

- `zoo::tools::ToolHandler`
- `zoo::tools::ToolDefinition`
- `zoo::tools::make_tool_definition(...)`
- typed callable `ToolRegistry::register_tool(...)` overloads
- JSON-handler `ToolRegistry::register_tool(...)` overloads
- `ToolRegistry::invoke(...)`
- `ToolRegistry::find_handler(...)`
- `ErrorCode::ToolExecutionFailed`
- `ErrorCode::InvalidToolSignature`

Schema-only registration:

```cpp
zoo::tools::ToolRegistry registry;
registry.register_tool("search", "Search local documents", schema);

model->set_tool_calling(registry.get_all_tool_specs());
```

### Hub

`zoo::hub::ModelStore::load_model(...)` now returns
`std::unique_ptr<zoo::Model>`. Catalog, import, and HuggingFace download APIs
remain under the optional `zoo::hub` namespace when `ZOO_BUILD_HUB=ON`.

### Compatibility Names

Compatibility-only aliases and forwarding surfaces are removed:

- `zoo::Message` -> `zoo::OwnedMessage`
- `zoo::ToolCallInfo` -> `zoo::OwnedToolCall`
- implicit `GenerationOverride` construction from `GenerationOptions`

Use `GenerationOverride::inherit_defaults()` to inherit configured defaults, or
`GenerationOverride::explicit_options(options)` to apply request options exactly.
