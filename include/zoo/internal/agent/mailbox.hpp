/**
 * @file mailbox.hpp
 * @brief Dual-lane mailbox for agent chat requests and control commands.
 */

#pragma once

#include "command.hpp"
#include "request.hpp"
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <variant>

namespace zoo::internal::agent {

/// A single item popped from the mailbox: either a chat request or a control command.
using WorkItem = std::variant<Request, Command>;

/**
 * @brief Thread-safe dual-lane mailbox for the agent runtime.
 *
 * The request lane is bounded by a configurable capacity. The command lane is
 * unbounded because control commands are rare and callers block on their result.
 * The pop order prioritizes pending commands over queued requests so that
 * model-affecting operations are applied between requests, never mid-generation.
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
     * @brief Enqueues a control command.
     *
     * Commands are always accepted unless the mailbox has been shut down.
     *
     * @return `true` when the command was accepted.
     */
    bool push_command(Command command) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            return false;
        }
        commands_.push(std::move(command));
        cv_.notify_one();
        return true;
    }

    /**
     * @brief Pops the next work item, blocking until one is available or shutdown.
     *
     * Pending commands are always dequeued before queued requests.
     *
     * @return The next work item, or `std::nullopt` after shutdown once both
     *         queues drain.
     */
    std::optional<WorkItem> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return !commands_.empty() || !requests_.empty() || shutdown_;
        });

        if (!commands_.empty()) {
            Command cmd = std::move(commands_.front());
            commands_.pop();
            return cmd;
        }

        if (!requests_.empty()) {
            Request req = std::move(requests_.front());
            requests_.pop();
            return req;
        }

        return std::nullopt;
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

    /// Returns the number of currently queued commands.
    size_t command_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return commands_.size();
    }

  private:
    std::queue<Request> requests_;
    std::queue<Command> commands_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t request_capacity_;
    bool shutdown_;
};

} // namespace zoo::internal::agent
