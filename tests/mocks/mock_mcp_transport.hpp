#pragma once

#include "zoo/mcp/transport/itransport.hpp"
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace zoo {
namespace testing {

/**
 * @brief Mock MCP transport for unit testing.
 *
 * Captures sent messages and allows injecting received messages.
 * No actual subprocess is spawned.
 */
class MockMcpTransport : public mcp::transport::ITransport {
public:
    // Configuration
    bool should_fail_connect = false;
    bool should_fail_send = false;
    std::string error_message = "Mock transport error";

    // State
    bool connected = false;
    std::vector<std::string> sent_messages;

    Expected<void> connect() override {
        if (should_fail_connect) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, error_message});
        }
        connected = true;
        return {};
    }

    void disconnect() override {
        connected = false;
    }

    bool is_connected() const override {
        return connected;
    }

    Expected<void> send(const std::string& message) override {
        if (!connected) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, "Not connected"});
        }
        if (should_fail_send) {
            return tl::unexpected(Error{ErrorCode::McpTransportFailed, error_message});
        }
        sent_messages.push_back(message);

        // Check if there's an auto-response queued
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (!queued_responses_.empty()) {
            std::string response = queued_responses_.front();
            queued_responses_.pop();
            // Deliver asynchronously via callback
            if (receive_callback_) {
                receive_callback_(response);
            }
        }

        return {};
    }

    void set_receive_callback(ReceiveCallback callback) override {
        receive_callback_ = std::move(callback);
    }

    void set_error_callback(ErrorCallback callback) override {
        error_callback_ = std::move(callback);
    }

    // ========================================================================
    // Test helpers
    // ========================================================================

    /**
     * @brief Queue a response to be auto-delivered after the next send().
     */
    void enqueue_response(const std::string& response) {
        std::lock_guard<std::mutex> lock(response_mutex_);
        queued_responses_.push(response);
    }

    /**
     * @brief Simulate receiving a message from the server.
     */
    void inject_message(const std::string& message) {
        if (receive_callback_) {
            receive_callback_(message);
        }
    }

    /**
     * @brief Simulate a transport error.
     */
    void inject_error(const std::string& error) {
        if (error_callback_) {
            error_callback_(error);
        }
    }

    /**
     * @brief Get the last sent message.
     */
    std::string last_sent() const {
        if (sent_messages.empty()) return "";
        return sent_messages.back();
    }

    /**
     * @brief Clear sent messages.
     */
    void clear_sent() {
        sent_messages.clear();
    }

private:
    ReceiveCallback receive_callback_;
    ErrorCallback error_callback_;
    std::mutex response_mutex_;
    std::queue<std::string> queued_responses_;
};

} // namespace testing
} // namespace zoo
