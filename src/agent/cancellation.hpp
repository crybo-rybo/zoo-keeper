/**
 * @file cancellation.hpp
 * @brief Shared cancellation signal for the agent runtime.
 *
 * `CancellationToken` is observed by every blocking call site inside the
 * inference thread (tool executor waits, command-mailbox waits, etc.) so that
 * `AgentRuntime::stop()` can return promptly even if a user tool handler is
 * still running.  A composite view combines the agent-wide token with the
 * per-request cancellation flag held on the request slot.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace zoo::internal::agent {

/// Single-shot cancellation signal owned by `AgentRuntime`.
class CancellationToken {
  public:
    [[nodiscard]] bool stopped() const noexcept {
        return stopped_.load(std::memory_order_acquire);
    }

    void request_stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }

    /// Sleeps up to `timeout` or until `request_stop()` is called.  Returns
    /// `true` if the token was signalled before the timeout expired.
    bool wait_for_stop(std::chrono::nanoseconds timeout) const {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, timeout, [this] { return stopped_.load(); });
    }

  private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::atomic<bool> stopped_{false};
};

/// Lightweight view used by per-request call sites to observe both the
/// agent-wide stop signal and the per-request cancellation flag.
class CompositeCancellation {
  public:
    constexpr CompositeCancellation(const CancellationToken& agent_stop,
                                    const std::atomic<bool>* request_cancelled) noexcept
        : agent_stop_(&agent_stop), request_cancelled_(request_cancelled) {}

    [[nodiscard]] bool cancelled() const noexcept {
        if (agent_stop_->stopped()) {
            return true;
        }
        return request_cancelled_ != nullptr && request_cancelled_->load(std::memory_order_acquire);
    }

    [[nodiscard]] const CancellationToken& agent_stop() const noexcept {
        return *agent_stop_;
    }

  private:
    const CancellationToken* agent_stop_;
    const std::atomic<bool>* request_cancelled_;
};

} // namespace zoo::internal::agent
