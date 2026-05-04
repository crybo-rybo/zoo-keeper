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
 * the dispatcher thread. For action-returning callbacks (`AsyncTokenCallback`
 * built from a callable that returns `TokenAction`), `dispatch()` blocks until
 * the dispatcher thread has executed the callback and returns its action. For
 * void-returning callbacks, `dispatch()` enqueues and returns
 * `TokenAction::Continue` immediately; any exception thrown by the callback is
 * captured and rethrown on the next `dispatch()` or `drain()` call. `drain()`
 * provides a synchronization point between generation passes.
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
     * @brief Enqueues a callback invocation for execution on the dispatcher thread.
     *
     * The token string is copied into the queue. The callback reference must
     * remain valid until `dispatch()` returns (action callbacks) or until the
     * next `drain()` returns (void callbacks).
     */
    TokenAction dispatch(AsyncTokenCallback& callback, std::string_view token) {
        if (callback.returns_action()) {
            return dispatch_sync(callback, token);
        }
        dispatch_async(callback, token);
        return TokenAction::Continue;
    }

    /**
     * @brief Blocks until all previously dispatched callbacks have completed.
     *
     * Rethrows any exception captured from a void-returning callback that ran
     * on the dispatcher thread since the last `drain()` or `dispatch()` call.
     */
    void drain() {
        std::exception_ptr captured;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            drain_cv_.wait(lock, [this] { return queue_.empty() && !executing_; });
            captured = std::exchange(failure_, nullptr);
        }
        if (captured) {
            std::rethrow_exception(captured);
        }
    }

  private:
    struct Entry {
        AsyncTokenCallback* callback;
        std::string token;
        // Set for action-returning callbacks; the inference thread waits on it.
        // Null for void-returning callbacks; exceptions are captured into
        // `failure_` and surfaced at the next `dispatch()`/`drain()`.
        std::shared_ptr<std::promise<TokenAction>> done;
    };

    TokenAction dispatch_sync(AsyncTokenCallback& callback, std::string_view token) {
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

    void dispatch_async(AsyncTokenCallback& callback, std::string_view token) {
        std::exception_ptr captured;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            captured = std::exchange(failure_, nullptr);
            if (!shutdown_) {
                queue_.push(Entry{&callback, std::string(token), nullptr});
            }
        }
        cv_.notify_one();
        if (captured) {
            std::rethrow_exception(captured);
        }
    }

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
                    TokenAction action = (*entry.callback)(entry.token);
                    if (entry.done) {
                        entry.done->set_value(action);
                    }
                } catch (const std::exception& e) {
                    ZOO_LOG("error", "streaming callback threw: %s", e.what());
                    if (entry.done) {
                        entry.done->set_exception(std::current_exception());
                    } else {
                        lock.lock();
                        if (!failure_) {
                            failure_ = std::current_exception();
                        }
                        lock.unlock();
                    }
                } catch (...) {
                    ZOO_LOG("error", "streaming callback threw unknown exception");
                    if (entry.done) {
                        entry.done->set_exception(std::current_exception());
                    } else {
                        lock.lock();
                        if (!failure_) {
                            failure_ = std::current_exception();
                        }
                        lock.unlock();
                    }
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
    std::exception_ptr failure_;
    std::thread thread_;
};

} // namespace zoo::internal::agent
