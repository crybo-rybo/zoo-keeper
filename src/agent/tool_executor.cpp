/**
 * @file tool_executor.cpp
 * @brief Off-thread tool handler execution.
 */

#include "agent/tool_executor.hpp"

#include "log.hpp"

#include <exception>
#include <thread>
#include <utility>

namespace zoo::internal::agent {

namespace {

Expected<nlohmann::json> invoke_tool_handler(tools::ToolHandler& handler, nlohmann::json& args) {
    try {
        return handler(args);
    } catch (const std::exception& e) {
        ZOO_LOG("error", "tool handler threw: %s", e.what());
        return std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, std::string("Tool handler threw: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "tool handler threw unknown exception");
        return std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, "Tool handler threw unknown exception"});
    }
}

void defer_thread_join(std::thread worker) {
    if (!worker.joinable()) {
        return;
    }
    std::thread([worker = std::move(worker)]() mutable {
        worker.join();
    }).detach();
}

} // namespace

ToolExecutor::Handle::Handle(std::future<Expected<nlohmann::json>> future,
                             std::shared_ptr<JobControl> control) noexcept
    : future_(std::move(future)), control_(std::move(control)) {}

ToolExecutor::Handle::~Handle() {
    release_worker();
}

Expected<nlohmann::json> ToolExecutor::Handle::get() {
    auto result = future_.get();
    release_worker();
    return result;
}

void ToolExecutor::Handle::release_worker() {
    if (!control_) {
        return;
    }
    std::thread worker;
    {
        std::lock_guard lock(control_->mutex);
        worker = std::move(control_->worker);
    }
    defer_thread_join(std::move(worker));
}

void ToolExecutor::Handle::abandon() {
    if (!control_) {
        return;
    }
    {
        std::lock_guard lock(control_->mutex);
        control_->abandoned = true;
    }
    release_worker();
}

ToolExecutor::ToolExecutor() = default;

ToolExecutor::~ToolExecutor() {
    shutdown();
}

ToolExecutor::Handle ToolExecutor::submit(tools::ToolHandler handler, nlohmann::json args) {
    auto promise = std::make_shared<std::promise<Expected<nlohmann::json>>>();
    auto future = promise->get_future();
    if (shutdown_.load(std::memory_order_acquire)) {
        promise->set_value(
            std::unexpected(Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
        return Handle(std::move(future), {});
    }

    auto control = std::make_shared<JobControl>();
    try {
        control->worker = std::thread([control, promise, handler = std::move(handler),
                                       args = std::move(args)]() mutable {
            auto result = invoke_tool_handler(handler, args);
            {
                std::lock_guard lock(control->mutex);
                if (control->abandoned) {
                    return;
                }
            }
            promise->set_value(std::move(result));
        });
    } catch (const std::exception& e) {
        ZOO_LOG("error", "failed to start tool handler thread: %s", e.what());
        promise->set_value(std::unexpected(
            Error{ErrorCode::ToolExecutionFailed,
                  std::string("Failed to start tool handler thread: ") + e.what()}));
    } catch (...) {
        ZOO_LOG("error", "failed to start tool handler thread");
        promise->set_value(std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, "Failed to start tool handler thread"}));
    }
    return Handle(std::move(future), std::move(control));
}

void ToolExecutor::shutdown() noexcept {
    shutdown_.store(true, std::memory_order_release);
}

} // namespace zoo::internal::agent
