/**
 * @file tool_executor.hpp
 * @brief Offloads tool handler invocations away from the inference thread.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/tools/types.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Executes tool handlers off the inference thread.
 *
 * Each submitted handler owns its callable, arguments, and promise. The caller can abandon the
 * returned handle during cancellation or shutdown without waiting for user code that may be blocked
 * indefinitely. Handler threads are joined asynchronously after completion or abandonment.
 */
class ToolExecutor {
  private:
    struct JobControl;

  public:
    class Handle {
      public:
        Handle() = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&&) noexcept = default;
        Handle& operator=(Handle&&) noexcept = default;
        ~Handle();

        template <typename Rep, typename Period>
        [[nodiscard]] std::future_status
        wait_for(const std::chrono::duration<Rep, Period>& timeout) const {
            return future_.wait_for(timeout);
        }

        [[nodiscard]] Expected<nlohmann::json> get();
        void abandon();

      private:
        friend class ToolExecutor;

        Handle(std::future<Expected<nlohmann::json>> future,
               std::shared_ptr<JobControl> control) noexcept;

        void release_worker();

        std::future<Expected<nlohmann::json>> future_;
        std::shared_ptr<JobControl> control_;
    };

    ToolExecutor();
    ~ToolExecutor();

    ToolExecutor(const ToolExecutor&) = delete;
    ToolExecutor& operator=(const ToolExecutor&) = delete;
    ToolExecutor(ToolExecutor&&) = delete;
    ToolExecutor& operator=(ToolExecutor&&) = delete;

    /**
     * @brief Submits a tool handler for execution.
     *
     * Returns a handle that resolves to the handler's return value. If called after shutdown, the
     * handle resolves immediately with an error.
     */
    [[nodiscard]] Handle submit(tools::ToolHandler handler, nlohmann::json args);
    void shutdown() noexcept;

  private:
    struct JobControl {
        std::mutex mutex;
        bool abandoned = false;
        std::thread worker;
    };

    std::atomic<bool> shutdown_{false};
};

} // namespace zoo::internal::agent
