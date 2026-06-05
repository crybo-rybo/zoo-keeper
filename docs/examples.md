# Examples Guide

Zoo-Keeper ships runnable example programs under [`examples/`](../examples/).
That directory is the source of truth for end-to-end usage.

## Build

```bash
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

## Programs

| Binary | Topic |
|--------|-------|
| [`minimal_model`](../examples/minimal_model.cpp) | First `zoo::Model` session |
| [`demo_chat`](../examples/demo_chat.cpp) | Interactive retained-history chat, streaming, metrics |
| [`demo_extract`](../examples/demo_extract.cpp) | Stateful, stateless, and streaming `Model::extract()` |
| [`model_generate`](../examples/model_generate.cpp) | Minimal one-shot generation |
| [`error_handling`](../examples/error_handling.cpp) | `Expected` error paths |
| [`stream_cancel`](../examples/stream_cancel.cpp) | Streaming + cancellation callback |
| [`manual_tool_schema`](../examples/manual_tool_schema.cpp) | Tool schemas, native call parsing, validation |

## API Sketches

### Streaming

```cpp
auto on_token = [](std::string_view token) {
    std::cout << token << std::flush;
    return zoo::TokenAction::Continue;
};

auto response = model->generate("Write a haiku", {}, on_token);
```

### Stateless Completion

```cpp
const std::array<zoo::MessageView, 2> messages = {
    zoo::MessageView{zoo::Role::System, "Reply in three words."},
    zoo::MessageView{zoo::Role::User, "Say hello."},
};

auto response = model->complete(zoo::ConversationView{std::span<const zoo::MessageView>(messages)});
```

### Cancellation

```cpp
auto should_cancel = [&] { return stop_requested.load(); };
auto response = model->generate("Write a long essay.", {}, on_token, should_cancel);
```

### Structured Extraction

```cpp
auto response = model->extract(schema, "There are 7 apples on the shelf.");
```

### Tool Parsing

```cpp
model->set_tool_calling(specs);
model->add_message(zoo::OwnedMessage::user("Search docs for llama.cpp.").view());
auto generated = model->generate_from_history();
for (const auto& call : generated->tool_calls) {
    // validate and execute in application code
}
// generated is already committed; append tool results before another pass.
```

## See Also

- [Getting Started](getting-started.md)
- [Tools](tools.md)
- [Configuration](configuration.md)
- [Building](building.md)
