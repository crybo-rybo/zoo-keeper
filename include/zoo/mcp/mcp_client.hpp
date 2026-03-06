#pragma once

#include "types.hpp"
#include "protocol/session.hpp"
#include "transport/itransport.hpp"
#include "transport/stdio_transport.hpp"
#include "../engine/tool_registry.hpp"
#include "../types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace zoo {
namespace mcp {

/**
 * @brief High-level MCP client that bridges MCP servers to ToolRegistry.
 *
 * Manages the full lifecycle of an MCP connection:
 *   1. Create transport (stdio subprocess)
 *   2. Establish session (initialize handshake)
 *   3. Discover tools (tools/list)
 *   4. Register tools into ToolRegistry as wrapped handlers
 *   5. Forward tool calls via JSON-RPC (tools/call)
 *
 * Each McpClient corresponds to one MCP server connection.
 */
class McpClient : public std::enable_shared_from_this<McpClient> {
public:
    struct Config {
        std::string server_id;                                         ///< Unique identifier for this server (used as tool prefix)
        transport::StdioTransport::Config transport;                   ///< Transport configuration
        protocol::Session::Config session;                             ///< Session configuration
        bool prefix_tools = true;                                      ///< Prefix tool names with "mcp_<server_id>:"
        std::chrono::milliseconds tool_timeout = std::chrono::seconds(30); ///< Timeout for individual tool calls
    };

    /**
     * @brief Factory method — creates but does NOT connect.
     */
    static Expected<std::shared_ptr<McpClient>> create(const Config& config) {
        if (config.server_id.empty()) {
            return tl::unexpected(Error{
                ErrorCode::InvalidConfig,
                "MCP server_id cannot be empty"
            });
        }

        // Use the two-step approach since we need shared_from_this
        auto client = std::shared_ptr<McpClient>(new McpClient(config));
        return client;
    }

    ~McpClient() {
        disconnect();
    }

    // Non-copyable
    McpClient(const McpClient&) = delete;
    McpClient& operator=(const McpClient&) = delete;

    /**
     * @brief Connect to the MCP server and perform initialization handshake.
     */
    Expected<void> connect() {
        if (session_ && session_->get_state() == SessionState::Ready) {
            return tl::unexpected(Error{ErrorCode::McpSessionFailed, "Already connected"});
        }

        auto transport = std::make_shared<transport::StdioTransport>(config_.transport);
        session_ = std::make_unique<protocol::Session>(transport, config_.session);

        auto result = session_->initialize();
        if (!result) {
            session_.reset();
            return tl::unexpected(result.error());
        }

        return {};
    }

    /**
     * @brief Disconnect from the MCP server.
     */
    void disconnect() {
        if (session_) {
            session_->shutdown();
            session_.reset();
        }
    }

    /**
     * @brief Check if connected and session is ready.
     */
    bool is_connected() const {
        return session_ && session_->get_state() == SessionState::Ready;
    }

    /**
     * @brief Discover tools available on the server via tools/list.
     */
    Expected<std::vector<McpToolDefinition>> discover_tools() {
        if (!is_connected()) {
            return tl::unexpected(Error{ErrorCode::McpDisconnected, "Not connected"});
        }

        if (!session_->get_server_capabilities().tools) {
            return std::vector<McpToolDefinition>{};
        }

        auto future = session_->send_request("tools/list");
        auto result = future.get();

        if (!result) {
            return tl::unexpected(result.error());
        }

        std::vector<McpToolDefinition> tools;
        try {
            if (result->contains("tools") && (*result)["tools"].is_array()) {
                for (const auto& tool_json : (*result)["tools"]) {
                    if (!tool_json.is_object()) {
                        return tl::unexpected(Error{
                            ErrorCode::McpProtocolError,
                            "Malformed tools/list response: tool entry is not a JSON object"
                        });
                    }
                    McpToolDefinition def;
                    def.name = tool_json.value("name", "");
                    def.description = tool_json.value("description", "");
                    if (tool_json.contains("inputSchema")) {
                        def.inputSchema = tool_json["inputSchema"];
                    } else {
                        def.inputSchema = nlohmann::json{
                            {"type", "object"},
                            {"properties", nlohmann::json::object()},
                            {"required", nlohmann::json::array()}
                        };
                    }
                    tools.push_back(std::move(def));
                }
            }
        } catch (const nlohmann::json::exception& e) {
            return tl::unexpected(Error{
                ErrorCode::McpProtocolError,
                std::string("Failed to parse tools/list response: ") + e.what()
            });
        }

        discovered_tools_ = tools;
        tools_discovered_ = true;
        return tools;
    }

    /**
     * @brief Register all discovered tools into a ToolRegistry.
     *
     * Each MCP tool is wrapped as a ToolHandler lambda that forwards calls
     * over the transport via JSON-RPC tools/call.
     */
    Expected<void> register_tools_with(engine::ToolRegistry& registry) {
        if (!tools_discovered_) {
            auto discover_result = discover_tools();
            if (!discover_result) {
                return tl::unexpected(discover_result.error());
            }
        }

        for (const auto& tool : discovered_tools_) {
            std::string registered_name = make_tool_name(tool.name);
            auto handler = wrap_mcp_tool(tool);
            registry.register_tool(registered_name, tool.description,
                                   tool.inputSchema, std::move(handler));
        }

        return {};
    }

    /**
     * @brief Call a tool on the MCP server.
     *
     * @param name The tool name (without prefix)
     * @param args JSON arguments to pass to the tool
     */
    Expected<nlohmann::json> call_tool(const std::string& name, const nlohmann::json& args) {
        if (!is_connected()) {
            return tl::unexpected(Error{ErrorCode::McpDisconnected, "Not connected"});
        }

        nlohmann::json params = {
            {"name", name},
            {"arguments", args}
        };

        auto future = session_->send_request("tools/call", params);

        // Wait with timeout
        auto status = future.wait_for(config_.tool_timeout);
        if (status == std::future_status::timeout) {
            return tl::unexpected(Error{ErrorCode::McpTimeout, "Tool call timed out: " + name});
        }

        auto result = future.get();
        if (!result) {
            return tl::unexpected(result.error());
        }

        // Extract content from MCP tool result
        // MCP tools/call returns: { "content": [...], "isError": false }
        try {
            if (result->contains("isError") && (*result)["isError"].get<bool>()) {
                std::string error_text;
                if (result->contains("content") && (*result)["content"].is_array()) {
                    for (const auto& item : (*result)["content"]) {
                        if (item.contains("text")) {
                            error_text += item["text"].get<std::string>();
                        }
                    }
                }
                return tl::unexpected(Error{
                    ErrorCode::McpServerError,
                    "Tool returned error: " + error_text
                });
            }

            // Return the full result for the tool handler to process
            return *result;
        } catch (const nlohmann::json::exception& e) {
            return tl::unexpected(Error{
                ErrorCode::McpProtocolError,
                std::string("Failed to parse tool result: ") + e.what()
            });
        }
    }

    const Config& get_config() const { return config_; }
    const std::vector<McpToolDefinition>& get_discovered_tools() const { return discovered_tools_; }
    const std::string& get_server_id() const { return config_.server_id; }

private:
    explicit McpClient(const Config& config)
        : config_(config) {}

    std::string make_tool_name(const std::string& tool_name) const {
        if (config_.prefix_tools) {
            return "mcp_" + config_.server_id + ":" + tool_name;
        }
        return tool_name;
    }

    engine::ToolHandler wrap_mcp_tool(const McpToolDefinition& def) {
        auto weak_self = weak_from_this();
        std::string tool_name = def.name;
        return [weak_self, tool_name](const nlohmann::json& args) -> Expected<nlohmann::json> {
            auto self = weak_self.lock();
            if (!self) {
                return tl::unexpected(Error{ErrorCode::McpDisconnected, "MCP client has been destroyed"});
            }
            auto result = self->call_tool(tool_name, args);
            if (!result) {
                return tl::unexpected(result.error());
            }

            // Wrap result for ToolRegistry format
            return nlohmann::json{{"result", *result}};
        };
    }

    Config config_;
    std::unique_ptr<protocol::Session> session_;
    std::vector<McpToolDefinition> discovered_tools_;
    bool tools_discovered_ = false;
};

} // namespace mcp
} // namespace zoo
