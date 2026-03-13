/**
 * @file request_tracker.hpp
 * @brief Tracks request identifiers, promises, and cancellation flags for Agent.
 */

#pragma once

#include "request.hpp"
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace zoo::internal::agent {

/**
 * @brief Request payload plus the future returned to the caller.
 */
struct PreparedRequest {
    Request request;
    std::future<Expected<Response>> future;
};

/**
 * @brief Owns request ids and shared completion state for in-flight requests.
 */
class RequestTracker {
  public:
    /**
     * @brief Allocates a request id and shared state for one submitted request.
     */
    PreparedRequest
    prepare(Message message,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt) {
        auto promise = std::make_shared<std::promise<Expected<Response>>>();
        auto future = promise->get_future();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);
        RequestId id = next_request_id_.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.emplace(id, State{promise, cancelled});
        }

        return PreparedRequest{
            Request(std::move(message), std::move(callback), promise, id, cancelled),
            std::move(future)};
    }

    PreparedRequest
    prepare(std::vector<Message> messages, HistoryMode history_mode,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt) {
        auto promise = std::make_shared<std::promise<Expected<Response>>>();
        auto future = promise->get_future();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);
        RequestId id = next_request_id_.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            requests_.emplace(id, State{promise, cancelled});
        }

        return PreparedRequest{
            Request(std::move(messages), history_mode, std::move(callback), promise, id, cancelled),
            std::move(future)};
    }

    /**
     * @brief Requests cooperative cancellation for a tracked request.
     */
    void cancel(RequestId id) {
        std::shared_ptr<std::atomic<bool>> cancelled;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = requests_.find(id);
            if (it == requests_.end()) {
                return;
            }
            cancelled = it->second.cancelled;
        }

        cancelled->store(true, std::memory_order_release);
    }

    /**
     * @brief Removes a completed request from the tracking map.
     */
    void cleanup(RequestId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        requests_.erase(id);
    }

    /**
     * @brief Resolves a tracked request future with an error and removes it.
     *
     * @return `true` when the request was still tracked.
     */
    bool fail(RequestId id, Error error) {
        std::shared_ptr<std::promise<Expected<Response>>> promise;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = requests_.find(id);
            if (it == requests_.end()) {
                return false;
            }
            promise = it->second.promise;
            requests_.erase(it);
        }

        if (promise) {
            promise->set_value(std::unexpected(std::move(error)));
        }
        return true;
    }

    /**
     * @brief Fails all tracked requests with the given error.
     *
     * Resolves every outstanding promise and clears the tracking map.
     * Used during shutdown to catch requests that were prepared but never
     * reached the mailbox.
     */
    void fail_all(const Error& error) {
        std::unordered_map<RequestId, State> snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snapshot.swap(requests_);
        }
        for (auto& [id, state] : snapshot) {
            if (state.promise) {
                state.promise->set_value(std::unexpected(error));
            }
        }
    }

    /// Returns the number of tracked requests.
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.size();
    }

  private:
    struct State {
        std::shared_ptr<std::promise<Expected<Response>>> promise;
        std::shared_ptr<std::atomic<bool>> cancelled;
    };

    std::atomic<RequestId> next_request_id_{1};
    mutable std::mutex mutex_;
    std::unordered_map<RequestId, State> requests_;
};

} // namespace zoo::internal::agent
