/**
 * @file test_tool_executor.cpp
 * @brief Unit tests for off-thread tool handler execution.
 */

#include "agent/tool_executor.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;

namespace {

zoo::tools::ToolHandler make_add_handler() {
    return [](const nlohmann::json& args) -> zoo::Expected<nlohmann::json> {
        return nlohmann::json{{"sum", args.at("a").get<int>() + args.at("b").get<int>()}};
    };
}

} // namespace

TEST(ToolExecutorTest, SubmittedJobCompletes) {
    zoo::internal::agent::ToolExecutor executor;

    auto handle = executor.submit(make_add_handler(), nlohmann::json{{"a", 1}, {"b", 2}});
    ASSERT_EQ(handle.wait_for(3s), std::future_status::ready);
    const auto result = handle.get();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ((*result)["sum"].get<int>(), 3);
}

TEST(ToolExecutorTest, ShutdownRejectsNewSubmissions) {
    zoo::internal::agent::ToolExecutor executor;
    executor.shutdown();

    auto handle = executor.submit(make_add_handler(), nlohmann::json{{"a", 1}, {"b", 2}});
    ASSERT_EQ(handle.wait_for(1s), std::future_status::ready);
    const auto result = handle.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::AgentNotRunning);
}

TEST(ToolExecutorTest, DestructorDoesNotBlockOnSlowHandler) {
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();
    auto executor = std::make_unique<zoo::internal::agent::ToolExecutor>();

    (void)executor->submit(
        [entered, release_future](const nlohmann::json&) mutable -> zoo::Expected<nlohmann::json> {
            entered->set_value();
            release_future.wait();
            return nlohmann::json{{"ok", true}};
        },
        nlohmann::json{});
    ASSERT_EQ(entered_future.wait_for(3s), std::future_status::ready);

    auto destroy_future = std::async(std::launch::async, [&executor] { executor.reset(); });
    EXPECT_EQ(destroy_future.wait_for(1s), std::future_status::ready);
    release->set_value();
    ASSERT_EQ(destroy_future.wait_for(3s), std::future_status::ready);
}

TEST(ToolExecutorTest, AbandonedBlockedJobDoesNotStarveLaterSubmissions) {
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    zoo::internal::agent::ToolExecutor executor;
    auto blocked = executor.submit(
        [entered, release_future](const nlohmann::json&) mutable -> zoo::Expected<nlohmann::json> {
            entered->set_value();
            release_future.wait();
            return nlohmann::json{{"ok", true}};
        },
        nlohmann::json{});
    ASSERT_EQ(entered_future.wait_for(3s), std::future_status::ready);

    blocked.abandon();
    auto fast = executor.submit(make_add_handler(), nlohmann::json{{"a", 1}, {"b", 2}});
    ASSERT_EQ(fast.wait_for(3s), std::future_status::ready);
    const auto result = fast.get();
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ((*result)["sum"].get<int>(), 3);

    release->set_value();
}

TEST(ToolExecutorTest, AbandonedHandlerThreadIsJoined) {
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();
    auto finished = std::make_shared<std::atomic<bool>>(false);

    zoo::internal::agent::ToolExecutor executor;
    auto blocked = executor.submit(
        [entered, release_future,
         finished](const nlohmann::json&) mutable -> zoo::Expected<nlohmann::json> {
            entered->set_value();
            release_future.wait();
            finished->store(true, std::memory_order_release);
            return nlohmann::json{{"ok", true}};
        },
        nlohmann::json{});
    ASSERT_EQ(entered_future.wait_for(3s), std::future_status::ready);

    blocked.abandon();
    release->set_value();

    for (int attempt = 0; attempt < 100; ++attempt) {
        if (finished->load(std::memory_order_acquire)) {
            break;
        }
        std::this_thread::sleep_for(50ms);
    }
    EXPECT_TRUE(finished->load(std::memory_order_acquire));
}

TEST(ToolExecutorTest, HandlerExceptionMapsToToolExecutionFailed) {
    zoo::internal::agent::ToolExecutor executor;
    auto handle = executor.submit(
        [](const nlohmann::json&) -> zoo::Expected<nlohmann::json> {
            throw std::runtime_error("boom");
        },
        nlohmann::json{});
    ASSERT_EQ(handle.wait_for(3s), std::future_status::ready);
    const auto result = handle.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolExecutionFailed);
}
