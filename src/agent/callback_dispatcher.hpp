/**
 * @file callback_dispatcher.hpp
 * @brief Offloads streaming callbacks to a dedicated thread.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Dispatches streaming callbacks on a dedicated thread.
 *
 * The inference thread calls `dispatch()` to enqueue a callback invocation
 * without blocking on the user's callback. The dispatcher thread executes
 * callbacks in FIFO order. `drain()` blocks until all queued callbacks have
 * been executed, providing a synchronization point between generation passes.
 */
class CallbackDispatcher {
  public:
    CallbackDispatcher() : thread_([this] { run(); }) {}

    ~CallbackDispatcher() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    CallbackDispatcher(const CallbackDispatcher&) = delete;
    CallbackDispatcher& operator=(const CallbackDispatcher&) = delete;
    CallbackDispatcher(CallbackDispatcher&&) = delete;
    CallbackDispatcher& operator=(CallbackDispatcher&&) = delete;

    /**
     * @brief Enqueues a callback invocation for async execution.
     *
     * The token string is copied into the queue. The callback reference must
     * remain valid until `drain()` returns.
     */
    void dispatch(std::function<void(std::string_view)>& callback, std::string_view token) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(Entry{&callback, std::string(token)});
        }
        cv_.notify_one();
    }

    /**
     * @brief Blocks until all previously dispatched callbacks have executed.
     */
    void drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        drain_cv_.wait(lock, [this] { return queue_.empty() && !executing_; });
    }

  private:
    struct Entry {
        std::function<void(std::string_view)>* callback;
        std::string token;
    };

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });

            while (!queue_.empty()) {
                auto entry = std::move(queue_.front());
                queue_.pop();
                executing_ = true;
                lock.unlock();

                (*entry.callback)(entry.token);

                lock.lock();
                executing_ = false;
                drain_cv_.notify_all();
            }

            if (shutdown_) {
                return;
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable drain_cv_;
    std::queue<Entry> queue_;
    bool shutdown_ = false;
    bool executing_ = false;
    std::thread thread_;
};

} // namespace zoo::internal::agent
