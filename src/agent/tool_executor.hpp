/**
 * @file tool_executor.hpp
 * @brief Offloads tool handler invocations away from the inference thread.
 */

#pragma once

#include "log.hpp"
#include "zoo/core/types.hpp"
#include "zoo/tools/types.hpp"

#include <atomic>
#include <exception>
#include <future>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>

namespace zoo::internal::agent {

/**
 * @brief Executes tool handlers off the inference thread.
 *
 * Each submitted handler owns its callable, arguments, and promise. The caller
 * can abandon the returned future during cancellation or shutdown without
 * waiting for user code that may be blocked indefinitely.
 */
class ToolExecutor {
  public:
    ToolExecutor() = default;
    ~ToolExecutor() {
        shutdown_.store(true, std::memory_order_release);
    }

    ToolExecutor(const ToolExecutor&) = delete;
    ToolExecutor& operator=(const ToolExecutor&) = delete;
    ToolExecutor(ToolExecutor&&) = delete;
    ToolExecutor& operator=(ToolExecutor&&) = delete;

    /**
     * @brief Submits a tool handler for execution.
     *
     * Returns a future that resolves to the handler's return value. If called after shutdown or if
     * the worker cannot be started, the future resolves immediately with an error.
     */
    [[nodiscard]] std::future<Expected<nlohmann::json>> submit(tools::ToolHandler handler,
                                                               nlohmann::json args) {
        auto promise = std::make_shared<std::promise<Expected<nlohmann::json>>>();
        auto future = promise->get_future();
        if (shutdown_.load(std::memory_order_acquire)) {
            promise->set_value(
                std::unexpected(Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
            return future;
        }

        try {
            std::thread([handler = std::move(handler), args = std::move(args), promise]() mutable {
                try {
                    promise->set_value(handler(args));
                } catch (const std::exception& e) {
                    ZOO_LOG("error", "tool handler threw: %s", e.what());
                    promise->set_value(
                        std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                              std::string("Tool handler threw: ") + e.what()}));
                } catch (...) {
                    ZOO_LOG("error", "tool handler threw unknown exception");
                    promise->set_value(std::unexpected(Error{
                        ErrorCode::ToolExecutionFailed, "Tool handler threw unknown exception"}));
                }
            }).detach();
        } catch (const std::exception& e) {
            promise->set_value(std::unexpected(
                Error{ErrorCode::ToolExecutionFailed,
                      std::string("Failed to start tool handler thread: ") + e.what()}));
        }
        return future;
    }

  private:
    std::atomic<bool> shutdown_{false};
};

} // namespace zoo::internal::agent
