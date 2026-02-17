#pragma once

#include "../types.hpp"
#include "json_rpc.hpp"
#include "../../types.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <unordered_map>

namespace zoo {
namespace mcp {
namespace protocol {

/**
 * @brief Async request correlation for JSON-RPC.
 *
 * Maps outgoing request IDs to std::promise objects so that responses
 * from the transport can be routed back to the correct caller.
 * Thread-safe: send_request() is called from the inference thread,
 * route_response() is called from the transport read thread.
 */
class MessageRouter {
public:
    using NotificationHandler = std::function<void(const std::string&, const nlohmann::json&)>;

    explicit MessageRouter(std::chrono::milliseconds default_timeout = std::chrono::seconds(30))
        : default_timeout_(default_timeout) {}

    /**
     * @brief Create a pending request and return its future.
     *
     * Generates a unique integer ID, stores a promise, and returns the
     * request ID along with a future that will resolve when the response arrives.
     */
    std::pair<int, std::future<Expected<nlohmann::json>>> create_pending_request() {
        int id = next_id_.fetch_add(1, std::memory_order_relaxed);
        auto promise = std::make_shared<std::promise<Expected<nlohmann::json>>>();
        auto future = promise->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_[id] = std::move(promise);
        }

        return {id, std::move(future)};
    }

    /**
     * @brief Route an incoming JSON-RPC message (response or notification).
     *
     * Called by the transport read thread when a message arrives.
     * For responses, resolves the corresponding promise.
     * For notifications, invokes the registered handler.
     */
    void route_message(const std::string& raw_json) {
        auto decoded = JsonRpc::decode(raw_json);

        if (decoded.is_error()) {
            return; // Malformed message — silently drop
        }

        if (decoded.is_response()) {
            route_response(*decoded.response);
        } else if (decoded.is_notification()) {
            if (notification_handler_) {
                notification_handler_(decoded.request->method, decoded.request->params);
            }
        }
    }

    /**
     * @brief Route a decoded JSON-RPC response to its pending request.
     */
    void route_response(const JsonRpcResponse& response) {
        int id = 0;
        if (std::holds_alternative<int>(response.id)) {
            id = std::get<int>(response.id);
        } else {
            // String IDs not used by our client — ignore
            return;
        }

        std::shared_ptr<std::promise<Expected<nlohmann::json>>> promise;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_.find(id);
            if (it == pending_.end()) {
                return; // No pending request for this ID
            }
            promise = std::move(it->second);
            pending_.erase(it);
        }

        if (response.is_error()) {
            promise->set_value(tl::unexpected(Error{
                ErrorCode::McpProtocolError,
                "JSON-RPC error " + std::to_string(response.error->code) + ": " + response.error->message
            }));
        } else if (response.result.has_value()) {
            promise->set_value(*response.result);
        } else {
            promise->set_value(nlohmann::json{});
        }
    }

    /**
     * @brief Set a handler for incoming notifications (no ID).
     */
    void set_notification_handler(NotificationHandler handler) {
        notification_handler_ = std::move(handler);
    }

    /**
     * @brief Cancel all pending requests with an error.
     *
     * Called during shutdown to ensure no futures are left dangling.
     */
    void cancel_all(const std::string& reason = "Router shutting down") {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [id, promise] : pending_) {
            promise->set_value(tl::unexpected(Error{
                ErrorCode::McpDisconnected,
                reason
            }));
        }
        pending_.clear();
    }

    /**
     * @brief Get the number of pending (in-flight) requests.
     */
    size_t pending_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }

    /**
     * @brief Get the default timeout for requests.
     */
    std::chrono::milliseconds default_timeout() const {
        return default_timeout_;
    }

private:
    std::atomic<int> next_id_{1};
    mutable std::mutex mutex_;
    std::unordered_map<int, std::shared_ptr<std::promise<Expected<nlohmann::json>>>> pending_;
    NotificationHandler notification_handler_;
    std::chrono::milliseconds default_timeout_;
};

} // namespace protocol
} // namespace mcp
} // namespace zoo
