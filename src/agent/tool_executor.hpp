/**
 * @file tool_executor.hpp
 * @brief Offloads tool handler invocations to a dedicated worker thread.
 */

#pragma once

#include "cancellation.hpp"
#include "log.hpp"
#include "zoo/core/types.hpp"
#include "zoo/tools/types.hpp"

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Executes tool handlers on a dedicated worker thread.
 *
 * The inference thread calls submit() to hand off a handler invocation and
 * then waits on the returned future via `wait_for_result()`, which observes a
 * `CompositeCancellation` view of the agent-wide stop token and the per-request
 * cancellation flag. This keeps arbitrary user-supplied tool code off the
 * inference thread while preserving sequential tool-loop semantics.  If the
 * caller bails out via cancellation, the handler keeps running on the worker
 * thread until completion — the runtime's stop() does not interrupt it (we
 * cannot kill arbitrary C++ code) but is no longer gated by it.
 */
class ToolExecutor {
  public:
    ToolExecutor() : thread_([this] { run(); }) {}

    ~ToolExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    ToolExecutor(const ToolExecutor&) = delete;
    ToolExecutor& operator=(const ToolExecutor&) = delete;
    ToolExecutor(ToolExecutor&&) = delete;
    ToolExecutor& operator=(ToolExecutor&&) = delete;

    /**
     * @brief Submits a tool handler for execution on the worker thread.
     *
     * Returns a future that resolves to the handler's return value. If called
     * after shutdown, the future resolves immediately with AgentNotRunning.
     */
    [[nodiscard]] std::future<Expected<nlohmann::json>> submit(tools::ToolHandler handler,
                                                               nlohmann::json args) {
        auto promise = std::make_shared<std::promise<Expected<nlohmann::json>>>();
        auto future = promise->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutdown_) {
                promise->set_value(std::unexpected(
                    Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
                return future;
            }
            queue_.push(Job{std::move(handler), std::move(args), std::move(promise)});
        }
        cv_.notify_one();
        return future;
    }

    /**
     * @brief Waits on a submitted future while observing a cancellation view.
     *
     * If `cancel.cancelled()` becomes true while waiting, returns
     * `RequestCancelled` without waiting for the handler to finish; the handler
     * continues to completion on the worker thread, and its result is dropped.
     * Otherwise returns the handler's result once the future is ready.
     */
    [[nodiscard]] static Expected<nlohmann::json>
    wait_for_result(std::future<Expected<nlohmann::json>>& future,
                    const CompositeCancellation& cancel,
                    std::chrono::nanoseconds poll_interval = std::chrono::milliseconds(25)) {
        while (true) {
            if (cancel.cancelled()) {
                return std::unexpected(Error{ErrorCode::RequestCancelled,
                                             "Request cancelled while a tool handler was running"});
            }
            const auto status = future.wait_for(poll_interval);
            if (status == std::future_status::ready) {
                return future.get();
            }
        }
    }

  private:
    struct Job {
        tools::ToolHandler handler;
        nlohmann::json args;
        std::shared_ptr<std::promise<Expected<nlohmann::json>>> promise;
    };

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });

            while (!queue_.empty()) {
                auto job = std::move(queue_.front());
                queue_.pop();
                lock.unlock();

                Expected<nlohmann::json> result;
                try {
                    result = job.handler(job.args);
                } catch (const std::exception& e) {
                    ZOO_LOG("error", "tool handler threw: %s", e.what());
                    result = std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                                   std::string("Tool handler threw: ") + e.what()});
                } catch (...) {
                    ZOO_LOG("error", "tool handler threw unknown exception");
                    result = std::unexpected(Error{ErrorCode::ToolExecutionFailed,
                                                   "Tool handler threw unknown exception"});
                }
                job.promise->set_value(std::move(result));

                lock.lock();
            }

            if (shutdown_) {
                return;
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Job> queue_;
    bool shutdown_ = false;
    std::thread thread_;
};

} // namespace zoo::internal::agent
