# Tool System

Zoo-Keeper provides a type-safe tool calling system that lets LLMs invoke native C++ functions during inference. The model detects when a tool should be called, the engine validates arguments, executes the function, and feeds the result back for continued generation.

## Template-Based Registration

Register any callable (function pointer, lambda, functor, `std::function`) with automatic JSON schema generation:

```cpp
// Free function
int add(int a, int b) { return a + b; }
agent->register_tool("add", "Add two integers", {"a", "b"}, add);

// Lambda
agent->register_tool("greet", "Greet a person", {"name"},
    [](std::string name) -> std::string {
        return "Hello, " + name + "!";
    });

// Zero-parameter function
std::string get_time() { return "2025-01-01 12:00:00"; }
agent->register_tool("get_time", "Get current time", {}, get_time);
```

### Supported Parameter Types

| C++ Type | JSON Schema Type |
|----------|-----------------|
| `int` | `integer` |
| `float` | `number` |
| `double` | `number` |
| `bool` | `boolean` |
| `std::string` | `string` |

The `param_names` vector must match the function's arity. A mismatch throws `std::invalid_argument`.

### Generated Schema

For `add(int a, int b)`, the registry generates:

```json
{
  "type": "function",
  "function": {
    "name": "add",
    "description": "Add two integers",
    "parameters": {
      "type": "object",
      "properties": {
        "a": { "type": "integer" },
        "b": { "type": "integer" }
      },
      "required": ["a", "b"]
    }
  }
}
```

## Manual Registration

For tools that need custom schemas beyond the supported types:

```cpp
nlohmann::json schema = {
    {"type", "object"},
    {"properties", {
        {"query", {{"type", "string"}}},
        {"limit", {{"type", "integer"}}}
    }},
    {"required", {"query"}}
};

zoo::engine::ToolHandler handler = [](const nlohmann::json& args)
    -> zoo::Expected<nlohmann::json> {
    std::string query = args.at("query").get<std::string>();
    int limit = args.value("limit", 10);
    // ... perform search ...
    return nlohmann::json{{"result", "found items"}};
};

// Access the registry directly for manual registration
registry.register_tool("search", "Search documents", schema, handler);
```

## Error Recovery and Retries

When the model produces invalid tool call arguments, the `ErrorRecovery` component:

1. Validates arguments against the registered JSON schema
2. Checks required fields and type correctness
3. On failure, injects an error message back into the conversation
4. The model gets another chance to self-correct (up to `max_retries`, default: 2)

If retries are exhausted, the request fails with `ErrorCode::ToolRetriesExhausted`.

## Tool Call Detection

The `ToolCallParser` scans model output for JSON objects containing `name` and `arguments` fields:

```json
{"name": "add", "arguments": {"a": 42, "b": 58}}
```

An optional `id` field is used for correlation; if absent, one is auto-generated (`call_1`, `call_2`, ...).

## Agentic Loop

The tool system integrates into the inference loop as follows:

1. Model generates text
2. Parser checks for a tool call in the output
3. If found: validate args, execute handler, inject result as a `Tool` message
4. Loop back for the model to process the tool result
5. Repeat until no tool call is detected or the loop limit is reached (default: 5 iterations)

Tool call and result history is captured in `Response::tool_calls`.

## Tool Call History

After a chat request completes, inspect which tools were called:

```cpp
auto response = agent->chat(zoo::Message::user("What is 42 + 58?")).get();
if (response) {
    for (const auto& msg : response->tool_calls) {
        std::cout << msg.content << std::endl;
    }
}
```

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 500 | `ToolNotFound` | Requested tool not in registry |
| 501 | `ToolExecutionFailed` | Handler threw or returned error |
| 502 | `InvalidToolSignature` | Signature doesn't match supported types |
| 503 | `ToolRetriesExhausted` | Max retries exceeded |
| 504 | `ToolLoopLimitReached` | Max agentic loop iterations exceeded |

## See Also

- [Getting Started](getting-started.md) -- basic Agent setup
- [Examples](examples.md) -- complete tool usage snippets
- [Architecture](architecture.md) -- how the agentic loop works internally
