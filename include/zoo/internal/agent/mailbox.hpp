/**
 * @file mailbox.hpp
 * @brief Bounded request mailbox for the agent runtime.
 */

#pragma once

#include "request.hpp"
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>

namespace zoo::internal::agent {

/**
 * @brief Thread-safe FIFO mailbox for queued chat requests.
 */
class RuntimeMailbox {
  public:
    explicit RuntimeMailbox(size_t request_capacity = 0)
        : request_capacity_(request_capacity), shutdown_(false) {}

    /**
     * @brief Enqueues one request when capacity and shutdown state permit it.
     *
     * @return `true` when the request was accepted.
     */
    bool push_request(Request request) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutdown_) {
            return false;
        }
        if (request_capacity_ > 0 && requests_.size() >= request_capacity_) {
            return false;
        }

        requests_.push(std::move(request));
        cv_.notify_one();
        return true;
    }

    /**
     * @brief Pops the next queued request, blocking until work or shutdown arrives.
     *
     * @return The next request, or `std::nullopt` after shutdown once the queue drains.
     */
    std::optional<Request> pop_request() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !requests_.empty() || shutdown_; });
        if (shutdown_ && requests_.empty()) {
            return std::nullopt;
        }

        Request request = std::move(requests_.front());
        requests_.pop();
        return request;
    }

    /// Marks the mailbox closed and wakes blocked waiters.
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

    /// Returns the number of currently queued requests.
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.size();
    }

  private:
    std::queue<Request> requests_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t request_capacity_;
    bool shutdown_;
};

} // namespace zoo::internal::agent
