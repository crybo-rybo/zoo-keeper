/**
 * @file tool_executor.hpp
 * @brief Offloads tool handler invocations away from the inference thread.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/tools/types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Executes tool handlers off the inference thread using a bounded worker pool.
 *
 * Each submitted handler owns its callable, arguments, and promise. The caller
 * can abandon the returned future during cancellation or shutdown without
 * waiting for user code that may be blocked indefinitely.
 */
class ToolExecutor {
  private:
    struct JobControl;
    struct State;

  public:
    class Handle {
      public:
        Handle() = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&&) noexcept = default;
        Handle& operator=(Handle&&) noexcept = default;

        template <typename Rep, typename Period>
        [[nodiscard]] std::future_status
        wait_for(const std::chrono::duration<Rep, Period>& timeout) const {
            return future_.wait_for(timeout);
        }

        [[nodiscard]] Expected<nlohmann::json> get();
        void abandon();

      private:
        friend class ToolExecutor;

        Handle(std::future<Expected<nlohmann::json>> future, std::shared_ptr<JobControl> control,
               std::weak_ptr<State> state) noexcept;

        std::future<Expected<nlohmann::json>> future_;
        std::shared_ptr<JobControl> control_;
        std::weak_ptr<State> state_;
    };

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
    [[nodiscard]] Handle submit(tools::ToolHandler handler, nlohmann::json args);

  private:
    struct JobControl {
        std::mutex mutex;
        bool running = false;
        bool abandoned = false;
        bool completed = false;
        bool replacement_started = false;
    };

    struct Job {
        tools::ToolHandler handler;
        nlohmann::json args;
        std::shared_ptr<std::promise<Expected<nlohmann::json>>> promise;
        std::shared_ptr<JobControl> control;
    };

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<Job> queue;
        std::atomic<bool> shutdown{false};
        size_t pooled_workers = 0;
    };
    // Shared by worker threads so detached workers can finish safely after ~ToolExecutor().

    static bool start_worker(const std::shared_ptr<State>& state) noexcept;
    static void worker_loop(const std::shared_ptr<State>& state);
    static void retire_worker(const std::shared_ptr<State>& state);
    static bool replace_abandoned_worker(const std::shared_ptr<State>& state,
                                         const std::shared_ptr<JobControl>& control) noexcept;
    static void fail_pending_jobs_locked(State& state, Error error);

    std::shared_ptr<State> state_;
};

} // namespace zoo::internal::agent
