# Tool System

Zoo-Keeper exposes one primary tool story: register tools on `zoo::Agent`, let the agent validate and execute them, and inspect the resulting `tool_invocations` after each request.

## Typed Registration

Register any supported callable and Zoo-Keeper will derive the argument schema automatically:

```cpp
int add(int a, int b) { return a + b; }

agent->register_tool("add", "Add two integers", {"a", "b"}, add);

agent->register_tool("greet", "Greet a person", {"name"},
    [](std::string name) -> std::string {
        return "Hello, " + name + "!";
    });

agent->register_tool("get_time", "Get current time", {}, []() -> std::string {
    return "2025-01-01 12:00:00";
});
```

Supported typed parameter types:

| C++ Type | JSON Schema Type |
|----------|------------------|
| `int` | `integer` |
| `float` | `number` |
| `double` | `number` |
| `bool` | `boolean` |
| `std::string` | `string` |

The parameter-name list must match the callable arity exactly or registration fails with `ErrorCode::InvalidToolSignature`.

## Manual Schema Registration

Use the manual path when you need a JSON-backed handler or schema features that typed registration does not express directly, such as optional parameters or enums.

```cpp
nlohmann::json schema = {
    {"type", "object"},
    {"properties", {
        {"query", {{"type", "string"}, {"description", "Search term"}}},
        {"limit", {{"type", "integer"}, {"enum", {5, 10, 20}}}},
        {"scope", {{"type", "string"}, {"enum", {"docs", "issues"}}}}
    }},
    {"required", {"query"}},
    {"additionalProperties", false}
};

auto result = agent->register_tool(
    "search_documents",
    "Search a local knowledge base.",
    schema,
    [](const nlohmann::json& args) -> zoo::Expected<nlohmann::json> {
        return nlohmann::json{
            {"query", args.at("query")},
            {"limit", args.value("limit", 10)},
            {"scope", args.value("scope", "docs")}
        };
    });
```

Manual handlers must accept a single JSON argument object and return `zoo::Expected<nlohmann::json>`.

## Supported Manual Schema Subset

Manual registration accepts a deliberately small subset of JSON Schema. Unsupported constructs fail fast during registration with `ErrorCode::InvalidToolSchema`.

Supported:

- top-level `"type": "object"`
- `"properties"` object
- primitive property types: `string`, `integer`, `number`, `boolean`
- `"required"` array
- property `"description"`
- property `"enum"`
- `"additionalProperties": false` or omission

Not supported:

- nested objects
- arrays and `items`
- `oneOf`, `anyOf`, `allOf`, `not`
- `$ref`
- numeric or string bounds such as `minimum`, `maximum`, `pattern`, `minLength`, `maxLength`
- unknown keywords that would change validation semantics

The runtime normalizes supported schemas into one internal representation and uses that same representation for:

- validation,
- deterministic schema export,
- grammar-constrained tool calling.

The same schema subset is accepted by `Agent::extract()` for grammar-constrained structured output. See [Structured Output](extract.md) for details.

## Validation and Retries

Every detected tool call is validated against the normalized registered schema, including grammar-constrained calls.

Validation enforces:

- required arguments are present,
- argument types match the registered primitive type,
- enum values match exactly when configured,
- unknown arguments are rejected.

If validation fails, the agent injects a corrective tool message and gives the model another chance to repair the call up to `Config::max_tool_retries`. Exhaustion fails the request with `ErrorCode::ToolRetriesExhausted`.

## Deterministic Ordering

Tool ordering is deterministic and follows registration order. That order is used for:

- `ToolRegistry::get_tool_names()`
- `ToolRegistry::get_all_schemas()`
- tool grammar generation
- tool listings embedded in the system prompt

Re-registering an existing tool updates it in place without moving its slot.

## Tool Invocation Records

After a chat request completes, inspect `Response::tool_invocations` to see what happened during the tool loop:

```cpp
auto handle = agent->chat(zoo::Message::user("What is 42 + 58?"));
auto response = handle.future.get();

if (response) {
    for (const auto& invocation : response->tool_invocations) {
        std::cout << invocation.name << " " << invocation.arguments_json << std::endl;

        if (invocation.result_json) {
            std::cout << *invocation.result_json << std::endl;
        }

        if (invocation.error) {
            std::cerr << invocation.error->to_string() << std::endl;
        }
    }
}
```

`ToolInvocationStatus` distinguishes:

- `Succeeded`
- `ValidationFailed`
- `ExecutionFailed`

## Low-Level Registry Access

`zoo::tools::ToolRegistry` remains public for lower-level usage, testing, or embedding inside custom runtimes. It is no longer the primary user path documented for normal application code.

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 500 | `ToolNotFound` | Requested tool name is not registered |
| 501 | `ToolExecutionFailed` | Handler threw or returned an execution failure |
| 502 | `InvalidToolSignature` | Typed registration metadata does not match the callable |
| 503 | `ToolRetriesExhausted` | Validation retry budget was exhausted |
| 504 | `ToolLoopLimitReached` | Agent exceeded the configured tool-iteration budget |
| 505 | `InvalidToolSchema` | Manual schema uses an unsupported construct |
| 506 | `ToolValidationFailed` | Parsed arguments failed validation |

## See Also

- [Getting Started](getting-started.md) -- basic Agent setup
- [Structured Output](extract.md) -- grammar-constrained extraction using the same schema subset
- [Examples](examples.md) -- complete usage snippets
- [Architecture](architecture.md) -- runtime structure and threading model
