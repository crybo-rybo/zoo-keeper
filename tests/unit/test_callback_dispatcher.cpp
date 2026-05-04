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
using zoo::AsyncTokenCallback;
using zoo::TokenAction;
using zoo::internal::agent::CallbackDispatcher;

TEST(CallbackDispatcherTest, DispatchedCallbacksArriveInOrder) {
    CallbackDispatcher dispatcher;

    std::mutex mutex;
    std::vector<std::string> received;

    AsyncTokenCallback callback = [&](std::string_view token) {
        std::lock_guard<std::mutex> lock(mutex);
        received.emplace_back(token);
    };

    EXPECT_EQ(dispatcher.dispatch(callback, "hello "), TokenAction::Continue);
    EXPECT_EQ(dispatcher.dispatch(callback, "world"), TokenAction::Continue);
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

    AsyncTokenCallback callback = [&](std::string_view) {
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
    AsyncTokenCallback callback = [&](std::string_view) {
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
    AsyncTokenCallback callback = [&](std::string_view) {
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

TEST(CallbackDispatcherTest, DispatchReturnsStopAction) {
    CallbackDispatcher dispatcher;
    AsyncTokenCallback callback = [](std::string_view) { return TokenAction::Stop; };

    EXPECT_EQ(dispatcher.dispatch(callback, "stop"), TokenAction::Stop);
}

TEST(CallbackDispatcherTest, ThrowingCallbackDoesNotTerminateProcess) {
    ASSERT_EXIT(
        [] {
            CallbackDispatcher dispatcher;

            AsyncTokenCallback failing = [](std::string_view) { throw std::runtime_error("boom"); };
            try {
                dispatcher.dispatch(failing, "x");
            } catch (const std::runtime_error&) {
            }

            bool recovered = false;
            AsyncTokenCallback succeeding = [&](std::string_view) { recovered = true; };
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

TEST(CallbackDispatcherTest, ThrowingCallbackPropagatesToDispatchAndDoesNotPoisonQueue) {
    CallbackDispatcher dispatcher;

    std::atomic<int> attempts{0};

    AsyncTokenCallback failing = [&](std::string_view) {
        attempts.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("boom");
    };

    EXPECT_THROW(dispatcher.dispatch(failing, "first"), std::runtime_error);

    EXPECT_EQ(attempts.load(std::memory_order_relaxed), 1);

    bool recovered = false;
    AsyncTokenCallback succeeding = [&](std::string_view) { recovered = true; };
    EXPECT_EQ(dispatcher.dispatch(succeeding, "next"), TokenAction::Continue);
    EXPECT_TRUE(recovered);
}

} // namespace
