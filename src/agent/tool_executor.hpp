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
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <thread>
#include <utility>

namespace zoo::internal::agent {

/**
 * @brief Executes tool handlers on a dedicated worker thread.
 *
 * The inference thread calls `submit()` to hand off a handler invocation, then
 * waits on the returned future via `wait_for_result()` while observing a
 * `CompositeCancellation` of the agent-wide stop token and the per-request
 * cancellation flag.
 *
 * Lifecycle. The worker thread holds a `shared_ptr<Shared>` covering every
 * piece of state it touches (mutex, queue, shutdown flag). The destructor
 * raises the shutdown flag, then **detaches** the worker thread instead of
 * joining it. Consequence: `~ToolExecutor` returns immediately even when a
 * user handler is still running, so `AgentRuntime` destruction never blocks
 * on a runaway handler. The handler runs to completion in the background; the
 * `Shared` state is cleaned up after the worker exits and drops its last
 * reference. Submitted promises are `shared_ptr` and stay alive whether or
 * not the original future is still being waited on, so the worker can always
 * publish (or simply discard) its result safely.
 */
class ToolExecutor {
  public:
    ToolExecutor() : shared_(std::make_shared<Shared>()) {
        std::thread worker([state = shared_] { run(state); });
        thread_ = std::move(worker);
    }

    ~ToolExecutor() {
        {
            std::lock_guard<std::mutex> lock(shared_->mutex);
            shared_->shutdown = true;
        }
        shared_->cv.notify_all();
        // Detach rather than join: the worker thread may currently be inside a
        // user handler that we cannot interrupt. The captured `state` shared_ptr
        // keeps the Shared block alive until the worker actually exits.
        if (thread_.joinable()) {
            thread_.detach();
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
            std::lock_guard<std::mutex> lock(shared_->mutex);
            if (shared_->shutdown) {
                promise->set_value(std::unexpected(
                    Error{ErrorCode::AgentNotRunning, "Tool executor is shut down"}));
                return future;
            }
            shared_->queue.push(Job{std::move(handler), std::move(args), std::move(promise)});
        }
        shared_->cv.notify_one();
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

    struct Shared {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<Job> queue;
        bool shutdown = false;
    };

    static void run(std::shared_ptr<Shared> state) {
        std::unique_lock<std::mutex> lock(state->mutex);
        while (true) {
            state->cv.wait(lock, [&] { return state->shutdown || !state->queue.empty(); });

            while (!state->queue.empty()) {
                auto job = std::move(state->queue.front());
                state->queue.pop();
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
                // The promise is shared_ptr; setting the value is safe even if
                // the original future was abandoned by a cancelled caller.
                try {
                    job.promise->set_value(std::move(result));
                } catch (const std::future_error&) {
                    // Promise already satisfied or broken — nothing to do.
                }

                lock.lock();
            }

            if (state->shutdown) {
                return;
            }
        }
    }

    std::shared_ptr<Shared> shared_;
    std::thread thread_;
};

} // namespace zoo::internal::agent
