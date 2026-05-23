/**
 * @file tool_executor.hpp
 * @brief Offloads tool handler invocations away from the inference thread.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/tools/types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <thread>
#include <vector>

namespace zoo::internal::agent {

/**
 * @brief Executes tool handlers off the inference thread using a bounded worker pool.
 *
 * Each submitted handler owns its callable, arguments, and promise. The caller
 * can abandon the returned future during cancellation or shutdown without
 * waiting for user code that may be blocked indefinitely.
 */
class ToolExecutor {
  public:
    explicit ToolExecutor(size_t worker_count = 2);
    ~ToolExecutor();

    ToolExecutor(const ToolExecutor&) = delete;
    ToolExecutor& operator=(const ToolExecutor&) = delete;
    ToolExecutor(ToolExecutor&&) = delete;
    ToolExecutor& operator=(ToolExecutor&&) = delete;

    /**
     * @brief Submits a tool handler for execution.
     *
     * Returns a future that resolves to the handler's return value. If called after shutdown or if
     * no workers are available, the future resolves immediately with an error.
     */
    [[nodiscard]] std::future<Expected<nlohmann::json>> submit(tools::ToolHandler handler,
                                                               nlohmann::json args);

  private:
    struct Job {
        tools::ToolHandler handler;
        nlohmann::json args;
        std::shared_ptr<std::promise<Expected<nlohmann::json>>> promise;
    };

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<Job> queue;
        std::atomic<bool> shutdown{false};
    };
    // Shared by worker threads so detached workers can finish safely after ~ToolExecutor().

    static void worker_loop(const std::shared_ptr<State>& state);
    static void fail_pending_jobs_locked(State& state, Error error);

    std::shared_ptr<State> state_;
    std::vector<std::thread> workers_;
};

} // namespace zoo::internal::agent
