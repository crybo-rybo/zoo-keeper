#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace zoo {
namespace testing {
namespace mcp_fixtures {

// ============================================================================
// JSON-RPC Messages
// ============================================================================

inline std::string valid_request() {
    return R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05"},"id":1})";
}

inline std::string valid_response() {
    return R"({"jsonrpc":"2.0","result":{"protocolVersion":"2024-11-05"},"id":1})";
}

inline std::string valid_notification() {
    return R"({"jsonrpc":"2.0","method":"notifications/initialized"})";
}

inline std::string error_response() {
    return R"({"jsonrpc":"2.0","error":{"code":-32600,"message":"Invalid Request"},"id":1})";
}

inline std::string error_response_with_data() {
    return R"({"jsonrpc":"2.0","error":{"code":-32602,"message":"Invalid params","data":{"details":"missing field"}},"id":2})";
}

inline std::string malformed_json() {
    return R"({this is not valid json})";
}

inline std::string missing_jsonrpc() {
    return R"({"method":"test","id":1})";
}

inline std::string wrong_jsonrpc_version() {
    return R"({"jsonrpc":"1.0","method":"test","id":1})";
}

// ============================================================================
// MCP Initialize Messages
// ============================================================================

inline std::string initialize_response(int id = 1) {
    nlohmann::json j = {
        {"jsonrpc", "2.0"},
        {"result", {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", {
                {"tools", {{"listChanged", true}}},
                {"resources", {{"subscribe", true}, {"listChanged", false}}},
                {"prompts", {{"listChanged", false}}},
                {"logging", nlohmann::json::object()}
            }},
            {"serverInfo", {
                {"name", "test-server"},
                {"version", "1.0.0"}
            }}
        }},
        {"id", id}
    };
    return j.dump();
}

inline std::string initialize_response_minimal(int id = 1) {
    nlohmann::json j = {
        {"jsonrpc", "2.0"},
        {"result", {
            {"protocolVersion", "2024-11-05"},
            {"capabilities", nlohmann::json::object()},
            {"serverInfo", {
                {"name", "minimal-server"},
                {"version", "0.1.0"}
            }}
        }},
        {"id", id}
    };
    return j.dump();
}

// ============================================================================
// MCP Tools Messages
// ============================================================================

inline std::string tools_list_response(int id = 2) {
    nlohmann::json j = {
        {"jsonrpc", "2.0"},
        {"result", {
            {"tools", {
                {
                    {"name", "read_file"},
                    {"description", "Read a file from the filesystem"},
                    {"inputSchema", {
                        {"type", "object"},
                        {"properties", {
                            {"path", {{"type", "string"}, {"description", "File path to read"}}}
                        }},
                        {"required", {"path"}}
                    }}
                },
                {
                    {"name", "write_file"},
                    {"description", "Write content to a file"},
                    {"inputSchema", {
                        {"type", "object"},
                        {"properties", {
                            {"path", {{"type", "string"}, {"description", "File path"}}},
                            {"content", {{"type", "string"}, {"description", "Content to write"}}}
                        }},
                        {"required", {"path", "content"}}
                    }}
                }
            }}
        }},
        {"id", id}
    };
    return j.dump();
}

inline std::string tools_list_empty_response(int id = 2) {
    nlohmann::json j = {
        {"jsonrpc", "2.0"},
        {"result", {
            {"tools", nlohmann::json::array()}
        }},
        {"id", id}
    };
    return j.dump();
}

inline std::string tool_call_response(int id = 3) {
    nlohmann::json j = {
        {"jsonrpc", "2.0"},
        {"result", {
            {"content", {
                {{"type", "text"}, {"text", "Hello, World!"}}
            }},
            {"isError", false}
        }},
        {"id", id}
    };
    return j.dump();
}

inline std::string tool_call_error_response(int id = 3) {
    nlohmann::json j = {
        {"jsonrpc", "2.0"},
        {"result", {
            {"content", {
                {{"type", "text"}, {"text", "File not found: /nonexistent"}}
            }},
            {"isError", true}
        }},
        {"id", id}
    };
    return j.dump();
}

} // namespace mcp_fixtures
} // namespace testing
} // namespace zoo
