/**
 * @file callback_dispatcher.hpp
 * @brief Offloads streaming callbacks to a dedicated thread.
 */

#pragma once

#include "log.hpp"
#include "zoo/core/types.hpp"

#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>

namespace zoo::internal::agent {

/**
 * @brief Dispatches streaming callbacks on a dedicated thread.
 *
 * The inference thread calls `dispatch()` to enqueue a callback invocation on
 * the dispatcher thread. `dispatch()` returns the callback's TokenAction after
 * the dispatcher thread executes it, and `drain()` provides a synchronization
 * point between generation passes.
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
     * remain valid until `dispatch()` returns.
     */
    TokenAction dispatch(AsyncTokenCallback& callback, std::string_view token) {
        auto done = std::make_shared<std::promise<TokenAction>>();
        auto future = done->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutdown_) {
                return TokenAction::Continue;
            }
            queue_.push(Entry{&callback, std::string(token), std::move(done)});
        }
        cv_.notify_one();
        return future.get();
    }

    /**
     * @brief Blocks until all previously dispatched callbacks have completed or been skipped.
     */
    void drain() {
        std::unique_lock<std::mutex> lock(mutex_);
        drain_cv_.wait(lock, [this] { return queue_.empty() && !executing_; });
    }

  private:
    struct Entry {
        AsyncTokenCallback* callback;
        std::string token;
        std::shared_ptr<std::promise<TokenAction>> done;
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

                try {
                    entry.done->set_value((*entry.callback)(entry.token));
                } catch (const std::exception& e) {
                    ZOO_LOG("error", "streaming callback threw: %s", e.what());
                    entry.done->set_exception(std::current_exception());
                } catch (...) {
                    ZOO_LOG("error", "streaming callback threw unknown exception");
                    entry.done->set_exception(std::current_exception());
                }

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
