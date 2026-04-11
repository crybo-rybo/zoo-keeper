/**
 * @file test_callback_dispatcher.cpp
 * @brief Unit tests for the async callback dispatcher.
 */

#include "agent/callback_dispatcher.hpp"
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using zoo::internal::agent::CallbackDispatcher;

TEST(CallbackDispatcherTest, DispatchedCallbacksArriveInOrder) {
    CallbackDispatcher dispatcher;

    std::mutex mutex;
    std::vector<std::string> received;

    std::function<void(std::string_view)> callback = [&](std::string_view token) {
        std::lock_guard<std::mutex> lock(mutex);
        received.emplace_back(token);
    };

    dispatcher.dispatch(callback, "hello ");
    dispatcher.dispatch(callback, "world");
    dispatcher.drain();

    std::lock_guard<std::mutex> lock(mutex);
    ASSERT_EQ(received.size(), 2u);
    EXPECT_EQ(received[0], "hello ");
    EXPECT_EQ(received[1], "world");
}

TEST(CallbackDispatcherTest, CallbacksRunOnDispatcherThread) {
    CallbackDispatcher dispatcher;

    std::thread::id callback_thread_id;
    std::promise<void> done;
    auto future = done.get_future();

    std::function<void(std::string_view)> callback = [&](std::string_view) {
        callback_thread_id = std::this_thread::get_id();
        done.set_value();
    };

    dispatcher.dispatch(callback, "test");
    ASSERT_EQ(future.wait_for(2s), std::future_status::ready);

    EXPECT_NE(callback_thread_id, std::thread::id{});
    EXPECT_NE(callback_thread_id, std::this_thread::get_id());
}

TEST(CallbackDispatcherTest, DrainBlocksUntilAllCallbacksComplete) {
    CallbackDispatcher dispatcher;

    std::atomic<int> count{0};
    std::function<void(std::string_view)> callback = [&](std::string_view) {
        std::this_thread::sleep_for(10ms);
        count.fetch_add(1, std::memory_order_relaxed);
    };

    for (int i = 0; i < 5; ++i) {
        dispatcher.dispatch(callback, "x");
    }
    dispatcher.drain();

    EXPECT_EQ(count.load(std::memory_order_relaxed), 5);
}

TEST(CallbackDispatcherTest, DestructorDrainsRemainingCallbacks) {
    std::atomic<int> count{0};
    std::function<void(std::string_view)> callback = [&](std::string_view) {
        count.fetch_add(1, std::memory_order_relaxed);
    };

    {
        CallbackDispatcher dispatcher;
        for (int i = 0; i < 3; ++i) {
            dispatcher.dispatch(callback, "x");
        }
    } // destructor should drain

    EXPECT_EQ(count.load(std::memory_order_relaxed), 3);
}

TEST(CallbackDispatcherTest, NoCallbackIsNoOp) {
    CallbackDispatcher dispatcher;
    dispatcher.drain(); // should not hang
}

TEST(CallbackDispatcherTest, ThrowingCallbackDoesNotTerminateProcess) {
    ASSERT_EXIT(
        [] {
            CallbackDispatcher dispatcher;

            std::function<void(std::string_view)> failing = [](std::string_view) {
                throw std::runtime_error("boom");
            };
            dispatcher.dispatch(failing, "x");

            try {
                dispatcher.drain();
            } catch (const std::exception&) {
            }

            bool recovered = false;
            std::function<void(std::string_view)> succeeding = [&](std::string_view) {
                recovered = true;
            };
            dispatcher.dispatch(succeeding, "y");

            try {
                dispatcher.drain();
            } catch (...) {
                std::exit(2);
            }

            std::exit(recovered ? 0 : 3);
        }(),
        ::testing::ExitedWithCode(0), "");
}

TEST(CallbackDispatcherTest, SkipsQueuedCallbacksAfterFirstFailureUntilDrain) {
    CallbackDispatcher dispatcher;

    std::promise<void> first_callback_entered;
    std::promise<void> release_first_callback;
    auto release_future = release_first_callback.get_future().share();
    std::atomic<int> attempts{0};

    std::function<void(std::string_view)> failing = [&](std::string_view) {
        const int invocation = attempts.fetch_add(1, std::memory_order_relaxed) + 1;
        if (invocation == 1) {
            first_callback_entered.set_value();
            release_future.wait();
        }
        throw std::runtime_error("boom");
    };

    dispatcher.dispatch(failing, "first");
    ASSERT_EQ(first_callback_entered.get_future().wait_for(2s), std::future_status::ready);

    for (int i = 0; i < 5; ++i) {
        dispatcher.dispatch(failing, "queued");
    }

    release_first_callback.set_value();

    EXPECT_THROW(dispatcher.drain(), std::runtime_error);
    EXPECT_EQ(attempts.load(std::memory_order_relaxed), 1);
}

} // namespace
