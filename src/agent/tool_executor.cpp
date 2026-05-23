/**
 * @file tool_executor.cpp
 * @brief Bounded worker pool for off-thread tool handler execution.
 */

#include "agent/tool_executor.hpp"

#include "log.hpp"

#include <exception>
#include <utility>

namespace zoo::internal::agent {

namespace {

void set_tool_handler_result(const std::shared_ptr<std::promise<Expected<nlohmann::json>>>& promise,
                             tools::ToolHandler& handler, nlohmann::json& args) {
    try {
        promise->set_value(handler(args));
    } catch (const std::exception& e) {
        ZOO_LOG("error", "tool handler threw: %s", e.what());
        promise->set_value(std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, std::string("Tool handler threw: ") + e.what()}));
    } catch (...) {
        ZOO_LOG("error", "tool handler threw unknown exception");
        promise->set_value(std::unexpected(
            Error{ErrorCode::ToolExecutionFailed, "Tool handler threw unknown exception"}));
    }
}

} // namespace

ToolExecutor::ToolExecutor(size_t worker_count) : state_(std::make_shared<State>()) {
    if (worker_count == 0) {
        state_->shutdown.store(true, std::memory_order_release);
        return;
    }

    workers_.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        try {
            workers_.emplace_back([state = state_]() { worker_loop(state); });
        } catch (const std::exception& e) {
            ZOO_LOG("error", "failed to start tool worker thread: %s", e.what());
            state_->shutdown.store(true, std::memory_order_release);
            break;
        }
    }
}

ToolExecutor::~ToolExecutor() {
    {
        std::lock_guard lock(state_->mutex);
        state_->shutdown.store(true, std::memory_order_release);
        fail_pending_jobs_locked(*state_,
                                 Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"});
    }
    state_->cv.notify_all();
    // Detach rather than join so agent shutdown never blocks on user tool handlers.
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.detach();
        }
    }
}

std::future<Expected<nlohmann::json>> ToolExecutor::submit(tools::ToolHandler handler,
                                                           nlohmann::json args) {
    auto promise = std::make_shared<std::promise<Expected<nlohmann::json>>>();
    auto future = promise->get_future();
    if (state_->shutdown.load(std::memory_order_acquire)) {
        promise->set_value(
            std::unexpected(Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
        return future;
    }

    {
        std::lock_guard lock(state_->mutex);
        if (state_->shutdown.load(std::memory_order_acquire)) {
            promise->set_value(
                std::unexpected(Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
            return future;
        }
        if (workers_.empty()) {
            promise->set_value(std::unexpected(
                Error{ErrorCode::ToolExecutionFailed, "Tool executor has no worker threads"}));
            return future;
        }
        state_->queue.push(Job{std::move(handler), std::move(args), std::move(promise)});
    }
    state_->cv.notify_one();
    return future;
}

void ToolExecutor::worker_loop(const std::shared_ptr<State>& state) {
    while (true) {
        Job job;
        {
            std::unique_lock lock(state->mutex);
            state->cv.wait(lock, [&state]() {
                return state->shutdown.load(std::memory_order_acquire) || !state->queue.empty();
            });
            if (state->shutdown.load(std::memory_order_acquire) && state->queue.empty()) {
                return;
            }
            job = std::move(state->queue.front());
            state->queue.pop();
        }

        set_tool_handler_result(job.promise, job.handler, job.args);
    }
}

void ToolExecutor::fail_pending_jobs_locked(State& state, Error error) {
    while (!state.queue.empty()) {
        auto job = std::move(state.queue.front());
        state.queue.pop();
        job.promise->set_value(std::unexpected(error));
    }
}

} // namespace zoo::internal::agent
