#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace zoo {
namespace mcp {

// ============================================================================
// JSON-RPC 2.0 Types
// ============================================================================

using RequestId = std::variant<int, std::string>;

struct JsonRpcError {
    int code = 0;
    std::string message;
    std::optional<nlohmann::json> data;

    bool operator==(const JsonRpcError& other) const {
        return code == other.code && message == other.message && data == other.data;
    }
};

struct JsonRpcRequest {
    std::string jsonrpc = "2.0";
    std::string method;
    nlohmann::json params = nlohmann::json::object();
    std::optional<RequestId> id;

    bool operator==(const JsonRpcRequest& other) const {
        return jsonrpc == other.jsonrpc && method == other.method &&
               params == other.params && id == other.id;
    }
};

struct JsonRpcResponse {
    std::string jsonrpc = "2.0";
    std::optional<nlohmann::json> result;
    std::optional<JsonRpcError> error;
    RequestId id = 0;

    bool is_error() const { return error.has_value(); }

    bool operator==(const JsonRpcResponse& other) const {
        return jsonrpc == other.jsonrpc && result == other.result &&
               error == other.error && id == other.id;
    }
};

// ============================================================================
// MCP Capability Types
// ============================================================================

struct ClientCapabilities {
    // Roots support (client can provide file roots to server)
    bool roots = false;
    bool roots_list_changed = false;

    // Sampling support (server can request LLM completions from client)
    bool sampling = false;
};

struct ServerCapabilities {
    // Tools support
    bool tools = false;
    bool tools_list_changed = false;

    // Resources support
    bool resources = false;
    bool resources_subscribe = false;
    bool resources_list_changed = false;

    // Prompts support
    bool prompts = false;
    bool prompts_list_changed = false;

    // Logging support
    bool logging = false;
};

// ============================================================================
// MCP Tool Types
// ============================================================================

struct McpToolDefinition {
    std::string name;
    std::string description;
    nlohmann::json inputSchema;

    bool operator==(const McpToolDefinition& other) const {
        return name == other.name && description == other.description &&
               inputSchema == other.inputSchema;
    }
};

// ============================================================================
// MCP Session Types
// ============================================================================

enum class SessionState {
    Disconnected,
    Connecting,
    Initializing,
    Ready,
    ShuttingDown
};

inline const char* session_state_to_string(SessionState state) {
    switch (state) {
        case SessionState::Disconnected: return "Disconnected";
        case SessionState::Connecting: return "Connecting";
        case SessionState::Initializing: return "Initializing";
        case SessionState::Ready: return "Ready";
        case SessionState::ShuttingDown: return "ShuttingDown";
    }
    return "unknown";
}

struct ServerInfo {
    std::string name;
    std::string version;
};

struct InitializeResult {
    std::string protocol_version;
    ServerCapabilities capabilities;
    ServerInfo server_info;
};

} // namespace mcp
} // namespace zoo
