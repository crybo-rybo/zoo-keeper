# Structured Output / Extract API

`Model::extract()` produces a JSON object whose shape is constrained at the
grammar level and returned as `ExtractionResponse::data`.

Runnable reference: [`examples/demo_extract.cpp`](../examples/demo_extract.cpp)

## Quick Start

```cpp
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

auto response = model->extract(schema, "Alice is 30 and scored 9.5.");

if (response) {
    std::cout << response->data["name"].get<std::string>() << '\n';
}
```

## API Reference

```cpp
Expected<ExtractionResponse> extract(
    const nlohmann::json& output_schema,
    std::string_view user_message,
    GenerationOverride generation = {},
    TokenCallback on_token = {},
    CancellationCallback should_cancel = {});

Expected<ExtractionResponse> extract(
    const nlohmann::json& output_schema,
    MessageView message,
    GenerationOverride generation = {},
    TokenCallback on_token = {},
    CancellationCallback should_cancel = {});

Expected<ExtractionResponse> extract(
    const nlohmann::json& output_schema,
    ConversationView messages,
    GenerationOverride generation = {},
    TokenCallback on_token = {},
    CancellationCallback should_cancel = {});
```

Schema validation happens before generation. Invalid schemas return
`ErrorCode::InvalidOutputSchema`.

## Stateful vs. Stateless

| | Stateful (`string_view` / `MessageView`) | Stateless (`ConversationView`) |
|-|----------------------|-------------------------------|
| History | User message and assistant JSON are committed | Retained history is restored afterward |
| System prompt | Active retained system prompt applies | Only the messages you provide |
| Use case | Ongoing conversations where extraction is one turn | Isolated extraction jobs, batch processing |

```cpp
const std::array<zoo::MessageView, 2> messages = {
    zoo::MessageView{zoo::Role::System, "Extract the structured entity described below."},
    zoo::MessageView{zoo::Role::User, "Bob is a 42-year-old engineer."},
};

auto response = model->extract(
    schema,
    zoo::ConversationView{std::span<const zoo::MessageView>(messages)});
```

## Streaming And Cancellation

```cpp
auto on_token = [&](std::string_view token) {
    streamed.append(token);
    return zoo::TokenAction::Continue;
};
auto should_cancel = [&] { return stop_requested.load(); };

auto response = model->extract(schema, "Carol is 25.", {}, on_token, should_cancel);
```

## Supported Schema Subset

`extract()` uses the same schema subset as manual tool registration:

- root type must be `"type": "object"`
- `"properties"` object with primitive-typed values
- primitive types: `string`, `integer`, `number`, `boolean`
- `"required"` array
- per-property `"enum"` constraints
- `"additionalProperties": false` or omission

Unsupported: nested objects, arrays, `oneOf`/`anyOf`/`allOf`, `$ref`, and
numeric/string bounds.

## How It Works

1. The JSON Schema is normalized into the same representation used by tool metadata.
2. A GBNF grammar rooted at a plain JSON object rule is generated.
3. The grammar is activated before the first generated token.
4. A single generation pass runs.
5. The raw output is parsed as JSON and validated against the schema.
6. The previous sampler/tool grammar state is restored.

## See Also

- [Tool System](tools.md)
- [Getting Started](getting-started.md)
- [Examples](../examples/README.md)
