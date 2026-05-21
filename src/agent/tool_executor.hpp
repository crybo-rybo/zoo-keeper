/**
 * @file tool_executor.hpp
 * @brief Offloads tool handler invocations to a detached worker thread.
 */

#pragma once

#include "log.hpp"
#include "zoo/core/types.hpp"
#include "zoo/tools/types.hpp"

#include <atomic>
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
 * State that outlives the executor lives in a `shared_ptr<Shared>` captured
 * by the worker. The destructor raises `shutdown` and detaches — it never
 * joins, so `~AgentRuntime` cannot deadlock on a runaway user handler. The
 * worker keeps Shared alive via its captured shared_ptr and exits naturally
 * once the current handler returns. Submitted promises are shared_ptr, so
 * publishing or discarding a result is safe regardless of whether the
 * caller's future was abandoned by cancellation.
 *
 * `wait_for_result()` polls a `running` flag (the agent-wide stop signal) and
 * an optional per-request `cancelled` flag, returning `RequestCancelled`
 * without waiting for the handler when either fires.
 */
class ToolExecutor {
  public:
    ToolExecutor() : shared_(std::make_shared<Shared>()) {
        std::thread(&ToolExecutor::run, shared_).detach();
    }

    ~ToolExecutor() {
        {
            std::lock_guard<std::mutex> lock(shared_->mutex);
            shared_->shutdown = true;
        }
        shared_->cv.notify_all();
    }

    ToolExecutor(const ToolExecutor&) = delete;
    ToolExecutor& operator=(const ToolExecutor&) = delete;
    ToolExecutor(ToolExecutor&&) = delete;
    ToolExecutor& operator=(ToolExecutor&&) = delete;

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

    /// Waits on `future` while polling cancellation flags. Returns
    /// `RequestCancelled` immediately if either flag fires; the handler keeps
    /// running on the worker thread and its result is dropped.
    [[nodiscard]] static Expected<nlohmann::json>
    wait_for_result(std::future<Expected<nlohmann::json>>& future, const std::atomic<bool>& running,
                    const std::atomic<bool>* request_cancelled,
                    std::chrono::nanoseconds poll = std::chrono::milliseconds(25)) {
        while (true) {
            if (!running.load(std::memory_order_acquire) ||
                (request_cancelled != nullptr &&
                 request_cancelled->load(std::memory_order_acquire))) {
                return std::unexpected(Error{ErrorCode::RequestCancelled,
                                             "Request cancelled while a tool handler was running"});
            }
            if (future.wait_for(poll) == std::future_status::ready) {
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
                job.promise->set_value(std::move(result));
                lock.lock();
            }
            if (state->shutdown) {
                return;
            }
        }
    }

    std::shared_ptr<Shared> shared_;
};

} // namespace zoo::internal::agent
