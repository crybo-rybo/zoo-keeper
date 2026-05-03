# Structured Output / Extract API

`Agent::extract()` produces a JSON object whose shape is constrained at the
grammar level and returned as `ExtractionResponse::data`.

The current API is response-first: start the request, call
`RequestHandle<ExtractionResponse>::await_result()`, and read the structured
payload from `response->data` if the request succeeds.

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

auto handle = agent->extract(schema, "Alice is 30 and scored 9.5.");
auto response = handle.await_result();

if (response) {
    std::cout << response->data["name"].get<std::string>() << '\n';
    std::cout << response->data["age"].get<int>() << '\n';
    std::cout << response->data["score"].get<double>() << '\n';
}
```

## API Reference

```cpp
// Stateful - append one string-like user message or MessageView to retained history
template <typename Message>
RequestHandle<ExtractionResponse> extract(
    const nlohmann::json& output_schema,
    Message&& message,
    const GenerationOptions& options = GenerationOptions{},
    AsyncTokenCallback callback = {});

// Stateless - use an explicit borrowed message sequence
RequestHandle<ExtractionResponse> extract(
    const nlohmann::json& output_schema,
    ConversationView messages,
    const GenerationOptions& options = GenerationOptions{},
    AsyncTokenCallback callback = {});
```

These entry points return a `RequestHandle<ExtractionResponse>`. The structured
result lives in `response->data`, and `response->tool_trace` is only populated
when `GenerationOptions::record_tool_trace` is enabled for the request.

Schema validation happens upfront on the calling thread before the request is
queued. An invalid schema is rejected immediately without waiting for the
inference thread.

## Stateful vs. Stateless

| | Stateful (`string_view` / `MessageView`) | Stateless (`ConversationView`) |
|-|----------------------|-------------------------------|
| History | User message appended; assistant output committed | Agent history untouched |
| System prompt | Active system prompt applied | Only the messages you provide |
| Use case | Ongoing conversations where extraction is one turn | Isolated extraction jobs, batch processing |

```cpp
// Stateless: full control over context; history is not affected
const std::array<zoo::MessageView, 2> messages = {
    zoo::MessageView{zoo::Role::System, "Extract the structured entity described below."},
    zoo::MessageView{zoo::Role::User, "Bob is a 42-year-old engineer."},
};
auto handle =
    agent->extract(schema, zoo::ConversationView{std::span<const zoo::MessageView>(messages)});
```

## Streaming

Pass an `on_token` callback to receive tokens as they are generated. The
callback runs on the CallbackDispatcher thread; avoid blocking inside it to
prevent backing up the dispatcher queue. Return `TokenAction::Stop` when the
callback has received enough output.

```cpp
auto handle = agent->extract(
    schema,
    "Carol is 25.",
    {},
    [](std::string_view token) {
        std::cout << token << std::flush;
        return zoo::TokenAction::Continue;
    });

auto response = handle.await_result();
```

## Cancellation

`RequestHandle::cancel()` cancels an in-progress extraction with the same
semantics as `chat()` cancellation. `Agent::cancel(handle.id())` remains
available when a caller needs to correlate requests externally. A cancelled
extraction returns `ErrorCode::RequestCancelled`.

```cpp
auto handle = agent->extract(schema, "...");
handle.cancel();
auto response = handle.await_result();
```

## Supported Schema Subset

`extract()` uses the same schema subset as [manual tool registration](tools.md#supported-manual-schema-subset):

- Root type must be `"type": "object"`
- `"properties"` object with primitive-typed values
- Primitive types: `string`, `integer`, `number`, `boolean`
- `"required"` array
- Per-property `"enum"` constraints
- `"additionalProperties": false` or omission

Not supported: nested objects, arrays, `oneOf`/`anyOf`/`allOf`, `$ref`,
numeric/string bounds.

Unsupported constructs are rejected upfront with `ErrorCode::InvalidOutputSchema`.

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 600 | `InvalidOutputSchema` | Schema is malformed or uses an unsupported construct - rejected before generation |
| 601 | `ExtractionFailed` | Generation succeeded but output failed JSON parsing or schema validation |

```cpp
auto response = handle.await_result();
if (!response) {
    switch (response.error().code) {
        case zoo::ErrorCode::InvalidOutputSchema:
            // bad schema - fix the schema definition
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

1. The JSON Schema is normalized into the same internal representation used by
   manual tool registration.
2. A GBNF grammar rooted at a plain JSON object rule is generated from the
   schema.
3. The grammar is activated immediately for the first generated token, so every
   sampled token is constrained to valid JSON matching the schema.
4. A single generation pass runs - no agentic tool loop.
5. The raw output is parsed as JSON and validated against the schema. On
   success, `response->data` holds the parsed object.
6. The previous tool grammar state is restored atomically, leaving the agent
   ready for the next request.

## See Also

- [Tool System](tools.md) -- register callable tools the model can invoke
- [Getting Started](getting-started.md) -- Agent setup and core API
- [Examples](examples.md) -- copy-paste code snippets
- [Architecture](architecture.md) -- runtime structure and threading model
