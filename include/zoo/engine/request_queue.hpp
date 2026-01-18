#pragma once

#include "../types.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

namespace zoo {
namespace engine {

/**
 * @brief Thread-safe MPSC queue for inference requests
 *
 * Multiple producer (calling threads), single consumer (inference thread).
 *
 * Design:
 * - Mutex + condition variable for simplicity and correctness
 * - Blocking pop for inference thread (wait for work)
 * - Non-blocking push for calling threads (return future immediately)
 * - Graceful shutdown support
 *
 * Future optimization: Lock-free MPSC queue if benchmarks show contention (NFR-03)
 */
class RequestQueue {
public:
    /**
     * @brief Construct queue with optional capacity limit
     *
     * @param max_size Maximum queue size (0 = unlimited)
     */
    explicit RequestQueue(size_t max_size = 0)
        : max_size_(max_size)
        , shutdown_(false)
    {}

    /**
     * @brief Push a request onto the queue (non-blocking)
     *
     * @param request Request to enqueue
     * @return true if enqueued, false if queue is full or shutdown
     */
    bool push(Request request) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Check shutdown
        if (shutdown_) {
            return false;
        }

        // Check capacity
        if (max_size_ > 0 && queue_.size() >= max_size_) {
            return false;  // Queue full
        }

        queue_.push(std::move(request));
        cv_.notify_one();
        return true;
    }

    /**
     * @brief Pop a request from the queue (blocking)
     *
     * Blocks until a request is available or shutdown is signaled.
     *
     * @return std::optional<Request> Request if available, nullopt if shutdown
     */
    std::optional<Request> pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for request or shutdown
        cv_.wait(lock, [this] {
            return !queue_.empty() || shutdown_;
        });

        // Check shutdown
        if (shutdown_ && queue_.empty()) {
            return std::nullopt;
        }

        // Pop request
        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

    /**
     * @brief Pop with timeout
     *
     * @param timeout Maximum time to wait
     * @return std::optional<Request> Request if available, nullopt if timeout or shutdown
     */
    template<typename Rep, typename Period>
    std::optional<Request> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait with timeout
        bool success = cv_.wait_for(lock, timeout, [this] {
            return !queue_.empty() || shutdown_;
        });

        if (!success || (shutdown_ && queue_.empty())) {
            return std::nullopt;
        }

        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

    /**
     * @brief Get current queue size
     *
     * @return size_t Number of pending requests
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * @brief Check if queue is empty
     *
     * @return bool True if no pending requests
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * @brief Signal shutdown
     *
     * Wakes up blocked pop() calls and prevents new pushes.
     * Does not clear existing requests.
     */
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

    /**
     * @brief Check if shutdown has been signaled
     *
     * @return bool True if shutdown
     */
    bool is_shutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return shutdown_;
    }

    /**
     * @brief Clear all pending requests
     *
     * Used for emergency cancellation.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
    }

private:
    std::queue<Request> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    bool shutdown_;
};

} // namespace engine
} // namespace zoo
