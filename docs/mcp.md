# MCP Client

Zoo-Keeper includes an optional MCP (Model Context Protocol) client that connects to external tool servers and federates their tools into the local ToolRegistry. MCP tools are invoked transparently during the agentic loop, just like natively registered tools.

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard for connecting AI models to external tools and data sources. An MCP server exposes a set of tools over a JSON-RPC 2.0 transport (typically stdio). Zoo-Keeper acts as an MCP client: it spawns the server process, negotiates capabilities, discovers available tools, and wraps each one as a callable entry in the ToolRegistry.

## Prerequisites

MCP support is optional and must be enabled at build time:

```bash
cmake -B build -DZOO_ENABLE_MCP=ON -DZOO_BUILD_TESTS=ON
cmake --build build
```

The `ZOO_ENABLE_MCP` flag pulls in the MCP protocol layer. When disabled, `add_mcp_server()` is still declared but returns `ErrorCode::McpTransportFailed`.

## Quick Start

```cpp
#include <zoo/zoo.hpp>

int main() {
    zoo::Config config;
    config.model_path = "models/llama-3-8b.gguf";
    config.context_size = 8192;

    auto agent = std::move(*zoo::Agent::create(config));
    agent->set_system_prompt("You are a helpful assistant with filesystem access.");

    // Connect to an MCP server
    zoo::mcp::McpClient::Config mcp;
    mcp.server_id = "filesystem";
    mcp.command = "npx";
    mcp.args = {"-y", "@modelcontextprotocol/server-filesystem", "/tmp"};

    auto result = agent->add_mcp_server(mcp);
    if (!result) {
        std::cerr << result.error().to_string() << std::endl;
        return 1;
    }

    // The model can now call filesystem tools
    auto response = agent->chat(zoo::Message::user("List files in /tmp")).get();
    if (response) {
        std::cout << response->text << std::endl;
    }
}
```

## McpClient::Config Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `server_id` | `std::string` | (required) | Unique identifier for this server connection |
| `command` | `std::string` | (required) | Executable to spawn (e.g. `"npx"`, `"python"`) |
| `args` | `std::vector<std::string>` | `{}` | Command-line arguments for the server process |
| `prefix_tools` | `bool` | `true` | Prefix tool names with `mcp_<server_id>:` |
| `tool_timeout` | `std::chrono::seconds` | `30s` | Timeout for individual tool call requests |

## How Tool Federation Works

When `add_mcp_server()` is called:

1. **Spawn** -- the transport layer starts the server process and connects via stdio pipes
2. **Initialize** -- a JSON-RPC `initialize` handshake negotiates protocol version and capabilities
3. **Discover** -- a `tools/list` request retrieves the server's tool definitions (name, description, JSON schema)
4. **Wrap** -- each discovered tool is wrapped in a `ToolHandler` that serializes arguments, sends a `tools/call` request over the transport, and deserializes the result
5. **Register** -- the wrapped handler is registered in the ToolRegistry with its schema

After registration, MCP tools participate in the agentic loop identically to local tools: the ToolCallParser detects them, ErrorRecovery validates arguments, and the AgenticLoop executes and injects results.

## Tool Name Prefixing

When `prefix_tools` is `true` (the default), tool names follow the pattern:

```
mcp_<server_id>:<tool_name>
```

For example, a tool named `read_file` from server `filesystem` becomes `mcp_filesystem:read_file`. This prevents collisions when multiple servers (or local tools) define tools with the same name.

Set `prefix_tools = false` if the server's tool names are already unique and you prefer shorter names.

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| 600 | `McpTransportFailed` | Transport layer failed (process spawn, pipe I/O) |
| 601 | `McpProtocolError` | JSON-RPC protocol violation |
| 602 | `McpServerError` | MCP server returned an error response |
| 603 | `McpSessionFailed` | Session initialization or capability negotiation failed |
| 604 | `McpToolNotAvailable` | Requested tool not available on the MCP server |
| 605 | `McpTimeout` | Request to MCP server timed out |
| 606 | `McpDisconnected` | MCP server disconnected unexpectedly |

## Threading Model

Each MCP server connection introduces one additional thread:

| Thread | Responsibility |
|--------|---------------|
| **MCP Transport Thread** (per server) | Reads stdout from the server process, parses JSON-RPC messages, routes responses via `MessageRouter` |

Tool call requests are sent from the inference thread (synchronously within the agentic loop). The transport thread receives the response and fulfills the pending future. The inference thread blocks on the future until the result arrives or the timeout expires.

`MessageRouter` uses a mutex-protected map of pending request IDs to promises, so response routing is thread-safe.

## See Also

- [Tools](tools.md) -- local tool registration and the agentic loop
- [Architecture](architecture.md) -- threading model and layer structure
- [Examples](examples.md) -- MCP usage snippets
