# Structured Output / Extract API

`Agent::extract()` generates a guaranteed-valid JSON object whose shape is constrained at the grammar level. Unlike prompting the model to "output JSON", the sampler physically cannot produce tokens that violate the schema — no post-processing retry loops required.

## Quick Start

```cpp
#include <zoo/zoo.hpp>

nlohmann::json schema = {
    {"type", "object"},
    {"properties", {
        {"name",  {{"type", "string"}}},
        {"age",   {{"type", "integer"}}},
        {"score", {{"type", "number"}}}
    }},
    {"required", {"name", "age"}},
    {"additionalProperties", false}
};

// Stateful: message is appended to agent history
auto handle = agent->extract(schema, zoo::Message::user("Alice is 30 and scored 9.5."));
auto response = handle.future.get();

if (response && response->extracted_data) {
    auto& data = *response->extracted_data;
    std::cout << data["name"].get<std::string>()    << '\n'; // Alice
    std::cout << data["age"].get<int>()              << '\n'; // 30
    std::cout << data["score"].get<double>()         << '\n'; // 9.5
}
```

## API Reference

```cpp
// Stateful — user message is appended to retained agent history
RequestHandle extract(
    const nlohmann::json& output_schema,
    Message message,
    std::optional<std::function<void(std::string_view)>> on_token = std::nullopt);

// Stateless — uses provided messages only; agent history is not modified
RequestHandle extract(
    const nlohmann::json& output_schema,
    std::vector<Message> messages,
    std::optional<std::function<void(std::string_view)>> on_token = std::nullopt);
```

Both overloads return a `RequestHandle` containing `.id` and `.future`. The structured result is available as `response->extracted_data` (`std::optional<nlohmann::json>`). Normal `chat()` responses always leave `extracted_data` as `std::nullopt`.

Schema validation happens **upfront on the calling thread** before the request is queued. An invalid schema is rejected immediately without waiting for the inference thread.

## Stateful vs. Stateless

| | Stateful (`Message`) | Stateless (`vector<Message>`) |
|-|----------------------|-------------------------------|
| History | User message appended; assistant output committed | Agent history untouched |
| System prompt | Active system prompt applied | Only the messages you provide |
| Use case | Ongoing conversations where extraction is one turn | Isolated extraction jobs, batch processing |

```cpp
// Stateless: full control over context; history is not affected
auto handle = agent->extract(schema, {
    zoo::Message::system("Extract the structured entity described below."),
    zoo::Message::user("Bob is a 42-year-old engineer.")
});
```

## Streaming

Pass an `on_token` callback to receive tokens as they are generated. The callback runs on the inference thread; avoid blocking inside it.

```cpp
auto handle = agent->extract(
    schema,
    zoo::Message::user("Carol is 25."),
    [](std::string_view token) {
        std::cout << token << std::flush;
    });

auto response = handle.future.get();
```

## Cancellation

`Agent::cancel(handle.id)` cancels an in-progress extraction with the same semantics as `chat()` cancellation. A cancelled extraction returns `ErrorCode::RequestCancelled`.

```cpp
auto handle = agent->extract(schema, zoo::Message::user("..."));
agent->cancel(handle.id);
auto response = handle.future.get(); // ErrorCode::RequestCancelled
```

## Supported Schema Subset

`extract()` uses the same schema subset as [manual tool registration](tools.md#supported-manual-schema-subset):

- Root type must be `"type": "object"`
- `"properties"` object with primitive-typed values
- Primitive types: `string`, `integer`, `number`, `boolean`
- `"required"` array
- Per-property `"enum"` constraints
- `"additionalProperties": false` or omission

Not supported: nested objects, arrays, `oneOf`/`anyOf`/`allOf`, `$ref`, numeric/string bounds.

Unsupported constructs are rejected upfront with `ErrorCode::InvalidOutputSchema`.

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 600 | `InvalidOutputSchema` | Schema is malformed or uses an unsupported construct — rejected before generation |
| 601 | `ExtractionFailed` | Generation succeeded but output failed JSON parsing or schema validation |

```cpp
auto response = handle.future.get();
if (!response) {
    switch (response.error().code) {
        case zoo::ErrorCode::InvalidOutputSchema:
            // bad schema — fix the schema definition
            break;
        case zoo::ErrorCode::ExtractionFailed:
            // model output was unparseable or type-mismatched
            break;
        default:
            break;
    }
}
```

## How It Works

1. The JSON Schema is normalized into an internal parameter list and validated — same path as manual tool registration.
2. A GBNF grammar rooted at a plain JSON object rule is generated from the schema (no `<tool_call>` sentinels, no tool-loop activation).
3. The grammar is activated immediately for the first generated token (non-lazy), so every sampled token is constrained to valid JSON matching the schema.
4. A single generation pass runs — no agentic tool loop.
5. The raw output is parsed as JSON and validated against the schema. On success, `response->extracted_data` holds the parsed object.
6. The previous tool grammar (if any) is restored atomically, leaving the agent ready for the next request.

## See Also

- [Tool System](tools.md) — register callable tools the model can invoke
- [Getting Started](getting-started.md) — Agent setup and core API
- [Examples](examples.md) — copy-paste code snippets
- [Architecture](architecture.md) — runtime structure and threading model
