#pragma once

#include "../types.hpp"
#include "../transport/itransport.hpp"
#include "json_rpc.hpp"
#include "message_router.hpp"
#include "../../types.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

namespace zoo {
namespace mcp {
namespace protocol {

/**
 * @brief MCP session state machine and RPC dispatch.
 *
 * Manages the MCP session lifecycle:
 *   Disconnected -> Connecting -> Initializing -> Ready -> ShuttingDown -> Disconnected
 *
 * Handles the 3-step initialization handshake:
 *   1. Client sends initialize request with capabilities
 *   2. Server responds with its capabilities
 *   3. Client sends initialized notification
 *
 * Threading: Session methods are called from the inference thread.
 * The transport's receive callback routes messages through the MessageRouter.
 */
class Session {
public:
    struct Config {
        std::string client_name = "zoo-keeper";
        std::string client_version = "0.1.0";
        std::string protocol_version = "2024-11-05";
        ClientCapabilities capabilities;
        std::chrono::milliseconds request_timeout = std::chrono::seconds(30);
    };

    explicit Session(std::shared_ptr<transport::ITransport> transport)
        : transport_(std::move(transport))
        , router_(config_.request_timeout) {
        // Wire transport receive to router
        transport_->set_receive_callback([this](const std::string& msg) {
            router_.route_message(msg);
        });
    }

    Session(std::shared_ptr<transport::ITransport> transport, Config config)
        : transport_(std::move(transport))
        , config_(std::move(config))
        , router_(config_.request_timeout) {
        // Wire transport receive to router
        transport_->set_receive_callback([this](const std::string& msg) {
            router_.route_message(msg);
        });
    }

    ~Session() {
        if (state_ == SessionState::Ready) {
            shutdown();
        }
    }

    /**
     * @brief Connect transport and perform MCP initialization handshake.
     *
     * Transitions: Disconnected -> Connecting -> Initializing -> Ready
     * On failure, transitions back to Disconnected.
     */
    Expected<void> initialize() {
        if (state_ != SessionState::Disconnected) {
            return tl::unexpected(Error{
                ErrorCode::McpSessionFailed,
                "Cannot initialize: session is in state " + std::string(session_state_to_string(state_))
            });
        }

        // Step 1: Connect transport
        state_ = SessionState::Connecting;
        auto connect_result = transport_->connect();
        if (!connect_result) {
            state_ = SessionState::Disconnected;
            return tl::unexpected(connect_result.error());
        }

        // Step 2: Send initialize request
        state_ = SessionState::Initializing;

        nlohmann::json init_params = {
            {"protocolVersion", config_.protocol_version},
            {"capabilities", build_client_capabilities()},
            {"clientInfo", {
                {"name", config_.client_name},
                {"version", config_.client_version}
            }}
        };

        auto init_result = send_request("initialize", init_params);
        auto future_result = init_result.get();

        if (!future_result) {
            state_ = SessionState::Disconnected;
            transport_->disconnect();
            return tl::unexpected(future_result.error());
        }

        // Parse server response
        auto parse_result = parse_initialize_result(*future_result);
        if (!parse_result) {
            state_ = SessionState::Disconnected;
            transport_->disconnect();
            return tl::unexpected(parse_result.error());
        }

        server_capabilities_ = parse_result->capabilities;
        server_info_ = parse_result->server_info;

        // Step 3: Send initialized notification
        auto notify_result = send_notification("notifications/initialized");
        if (!notify_result) {
            state_ = SessionState::Disconnected;
            transport_->disconnect();
            return tl::unexpected(notify_result.error());
        }

        state_ = SessionState::Ready;
        return {};
    }

    /**
     * @brief Gracefully shut down the session.
     */
    void shutdown() {
        if (state_ != SessionState::Ready && state_ != SessionState::Initializing) {
            return;
        }

        state_ = SessionState::ShuttingDown;

        // Cancel all pending requests
        router_.cancel_all("Session shutting down");

        // Disconnect transport
        transport_->disconnect();

        state_ = SessionState::Disconnected;
    }

    /**
     * @brief Send a JSON-RPC request and return a future for the result.
     */
    std::future<Expected<nlohmann::json>> send_request(const std::string& method,
                                                       const nlohmann::json& params = nlohmann::json::object()) {
        auto [id, future] = router_.create_pending_request();

        JsonRpcRequest request;
        request.method = method;
        request.params = params;
        request.id = id;

        auto encoded = JsonRpc::encode_request(request);
        auto send_result = transport_->send(encoded);

        if (!send_result) {
            // If send fails, we need to resolve the pending promise with an error
            // The router still has the promise — create a response to route the error
            JsonRpcResponse error_response;
            error_response.id = id;
            error_response.error = JsonRpcError{-32603, send_result.error().message, std::nullopt};
            router_.route_response(error_response);
        }

        return std::move(future);
    }

    /**
     * @brief Send a JSON-RPC notification (no response expected).
     */
    Expected<void> send_notification(const std::string& method,
                                     const nlohmann::json& params = nlohmann::json::object()) {
        auto encoded = JsonRpc::encode_notification(method, params);
        return transport_->send(encoded);
    }

    SessionState get_state() const { return state_; }
    const ServerCapabilities& get_server_capabilities() const { return server_capabilities_; }
    const ServerInfo& get_server_info() const { return server_info_; }
    MessageRouter& get_router() { return router_; }

private:
    nlohmann::json build_client_capabilities() const {
        nlohmann::json caps = nlohmann::json::object();

        if (config_.capabilities.roots) {
            nlohmann::json roots = nlohmann::json::object();
            if (config_.capabilities.roots_list_changed) {
                roots["listChanged"] = true;
            }
            caps["roots"] = roots;
        }

        if (config_.capabilities.sampling) {
            caps["sampling"] = nlohmann::json::object();
        }

        return caps;
    }

    Expected<InitializeResult> parse_initialize_result(const nlohmann::json& result) {
        InitializeResult init;

        try {
            init.protocol_version = result.value("protocolVersion", "");

            if (result.contains("serverInfo")) {
                init.server_info.name = result["serverInfo"].value("name", "");
                init.server_info.version = result["serverInfo"].value("version", "");
            }

            if (result.contains("capabilities")) {
                auto& caps = result["capabilities"];

                if (caps.contains("tools")) {
                    init.capabilities.tools = true;
                    if (caps["tools"].contains("listChanged")) {
                        init.capabilities.tools_list_changed = caps["tools"]["listChanged"].get<bool>();
                    }
                }

                if (caps.contains("resources")) {
                    init.capabilities.resources = true;
                    if (caps["resources"].contains("subscribe")) {
                        init.capabilities.resources_subscribe = caps["resources"]["subscribe"].get<bool>();
                    }
                    if (caps["resources"].contains("listChanged")) {
                        init.capabilities.resources_list_changed = caps["resources"]["listChanged"].get<bool>();
                    }
                }

                if (caps.contains("prompts")) {
                    init.capabilities.prompts = true;
                    if (caps["prompts"].contains("listChanged")) {
                        init.capabilities.prompts_list_changed = caps["prompts"]["listChanged"].get<bool>();
                    }
                }

                if (caps.contains("logging")) {
                    init.capabilities.logging = true;
                }
            }
        } catch (const nlohmann::json::exception& e) {
            return tl::unexpected(Error{
                ErrorCode::McpProtocolError,
                std::string("Failed to parse initialize result: ") + e.what()
            });
        }

        return init;
    }

    std::shared_ptr<transport::ITransport> transport_;
    Config config_;
    MessageRouter router_;

    SessionState state_ = SessionState::Disconnected;
    ServerCapabilities server_capabilities_;
    ServerInfo server_info_;
};

} // namespace protocol
} // namespace mcp
} // namespace zoo
