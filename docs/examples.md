# Examples Cookbook

Complete, copy-paste code snippets for common Zoo-Keeper patterns.

## Streaming Output

Print tokens as they arrive:

```cpp
auto future = agent->chat(
    zoo::Message::user("Write a haiku about AI"),
    [](std::string_view token) {
        std::cout << token << std::flush;
    }
);

auto response = future.get();
std::cout << std::endl;
```

## Multi-Turn Conversation

History is managed automatically:

```cpp
agent->chat(zoo::Message::user("My name is Alice")).get();
auto response = agent->chat(zoo::Message::user("What's my name?")).get();
// response->text will reference "Alice"
```

## Tool Registration and Calling

```cpp
// Define tools
int add(int a, int b) { return a + b; }
std::string get_time() {
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

// Register
agent->register_tool("add", "Add two integers", {"a", "b"}, add);
agent->register_tool("get_time", "Get current date and time", {}, get_time);

// The model can now call these tools
auto response = agent->chat(zoo::Message::user("What is 42 + 58?")).get();
if (response) {
    std::cout << response->text << std::endl;

    // Inspect tool call history
    for (const auto& msg : response->tool_calls) {
        std::cout << "Tool: " << msg.content << std::endl;
    }
}
```

## Lambda Tools

```cpp
agent->register_tool("uppercase", "Convert text to uppercase", {"text"},
    [](std::string text) -> std::string {
        std::transform(text.begin(), text.end(), text.begin(), ::toupper);
        return text;
    });
```

## RAG with InMemoryRagStore

```cpp
// Set up retriever
auto store = std::make_shared<zoo::engine::InMemoryRagStore>();
store->add_chunk({"c1", "Zoo-Keeper is a C++17 agent engine built on llama.cpp.", "docs"});
store->add_chunk({"c2", "It supports tool calling with automatic schema generation.", "docs"});
agent->set_retriever(store);

// Enable RAG per request
zoo::ChatOptions opts;
opts.rag.enabled = true;
opts.rag.top_k = 2;

auto response = agent->chat(
    zoo::Message::user("What is Zoo-Keeper?"), opts
).get();

if (response) {
    std::cout << response->text << std::endl;

    // Provenance: which chunks were used
    for (const auto& chunk : response->rag_chunks) {
        std::cout << "Used chunk: " << chunk.id
                  << " (score=" << chunk.score << ")" << std::endl;
    }
}
```

## RAG with Context Override

Inject your own context without a retriever:

```cpp
zoo::ChatOptions opts;
opts.rag.enabled = true;
opts.rag.context_override = "The capital of France is Paris. Population: 2.1 million.";

auto response = agent->chat(
    zoo::Message::user("What's the population of the French capital?"), opts
).get();
```

## Context Database (Long-Term Memory)

```cpp
// Enable SQLite memory
auto init = agent->enable_context_database("memory.sqlite");
if (!init) {
    std::cerr << init.error().to_string() << std::endl;
}

// Now chat normally -- old messages are automatically archived
// and retrieved when relevant
for (int i = 0; i < 100; ++i) {
    auto response = agent->chat(
        zoo::Message::user("Tell me fact #" + std::to_string(i))
    ).get();
}
```

## Error Handling

```cpp
auto result = agent->chat(zoo::Message::user("Hello")).get();

if (!result) {
    zoo::Error error = result.error();
    switch (error.code) {
        case zoo::ErrorCode::ContextWindowExceeded:
            std::cerr << "Context full!" << std::endl;
            agent->clear_history();
            break;
        case zoo::ErrorCode::InferenceFailed:
            std::cerr << "Inference failed: " << error.message << std::endl;
            break;
        case zoo::ErrorCode::ToolRetriesExhausted:
            std::cerr << "Tool retries exhausted: " << error.message << std::endl;
            break;
        default:
            std::cerr << error.to_string() << std::endl;
    }
} else {
    std::cout << result->text << std::endl;
}
```

## Cancellation

```cpp
auto future = agent->chat(zoo::Message::user("Write a long essay"));

// Cancel after 5 seconds
std::this_thread::sleep_for(std::chrono::seconds(5));
agent->stop();

auto result = future.get();
if (!result) {
    // result.error().code == ErrorCode::RequestCancelled
    std::cout << "Generation cancelled" << std::endl;
}
```

## Metrics

```cpp
auto response = agent->chat(zoo::Message::user("Hello")).get();
if (response) {
    std::cout << "Latency: " << response->metrics.latency_ms.count() << " ms" << std::endl;
    std::cout << "TTFT: " << response->metrics.time_to_first_token_ms.count() << " ms" << std::endl;
    std::cout << "Speed: " << response->metrics.tokens_per_second << " tok/s" << std::endl;
    std::cout << "Tokens: " << response->usage.prompt_tokens << " prompt + "
              << response->usage.completion_tokens << " completion" << std::endl;
}
```

## MCP Server Integration

Connect to an MCP-compatible tool server and use its tools transparently:

```cpp
// Configure an MCP server connection
zoo::mcp::McpClient::Config mcp_config;
mcp_config.server_id = "filesystem";
mcp_config.transport.command = "npx";
mcp_config.transport.args = {"-y", "@modelcontextprotocol/server-filesystem", "/tmp"};
mcp_config.prefix_tools = true;       // tools named mcp_filesystem:<tool_name>
mcp_config.tool_timeout = std::chrono::seconds(30);

// Connect -- discovers tools and registers them into the ToolRegistry
auto result = agent->add_mcp_server(mcp_config);
if (!result) {
    std::cerr << "MCP error: " << result.error().to_string() << std::endl;
}

// Now the model can call MCP tools just like local tools
auto response = agent->chat(
    zoo::Message::user("List files in /tmp")
).get();

if (response) {
    std::cout << response->text << std::endl;
}
```

Multiple MCP servers can be connected simultaneously:

```cpp
zoo::mcp::McpClient::Config git_config;
git_config.server_id = "git";
git_config.transport.command = "npx";
git_config.transport.args = {"-y", "@modelcontextprotocol/server-git"};
agent->add_mcp_server(git_config);

std::cout << "Connected servers: " << agent->mcp_server_count() << std::endl;
```

## See Also

- [Getting Started](getting-started.md) -- setup walkthrough
- [Tools](tools.md) -- tool system deep-dive
- [MCP](mcp.md) -- MCP client details
- [RAG Retrieval](rag.md) -- retrieval pipeline details
- [Configuration](configuration.md) -- all config options
