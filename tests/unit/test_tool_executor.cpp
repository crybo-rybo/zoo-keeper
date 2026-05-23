/**
 * @file test_tool_executor.cpp
 * @brief Unit tests for the bounded ToolExecutor worker pool.
 */

#include "agent/tool_executor.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

zoo::tools::ToolHandler make_add_handler() {
    return [](const nlohmann::json& args) -> zoo::Expected<nlohmann::json> {
        return nlohmann::json{{"sum", args.at("a").get<int>() + args.at("b").get<int>()}};
    };
}

} // namespace

TEST(ToolExecutorTest, ConcurrentJobsCompleteWithFixedWorkerPool) {
    zoo::internal::agent::ToolExecutor executor(2);

    std::vector<std::future<zoo::Expected<nlohmann::json>>> futures;
    futures.reserve(6);
    for (int i = 0; i < 6; ++i) {
        futures.push_back(executor.submit(make_add_handler(), nlohmann::json{{"a", i}, {"b", 1}}));
    }

    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ(futures[i].wait_for(3s), std::future_status::ready);
        const auto result = futures[i].get();
        ASSERT_TRUE(result.has_value()) << result.error().to_string();
        EXPECT_EQ((*result)["sum"].get<int>(), i + 1);
    }
}

TEST(ToolExecutorTest, ShutdownRejectsNewSubmissions) {
    zoo::internal::agent::ToolExecutor executor(0);

    auto future = executor.submit(make_add_handler(), nlohmann::json{{"a", 1}, {"b", 2}});
    ASSERT_EQ(future.wait_for(1s), std::future_status::ready);
    const auto result = future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::AgentNotRunning);
}

TEST(ToolExecutorTest, DestructorCompletesAfterActiveHandlerFinishes) {
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    {
        zoo::internal::agent::ToolExecutor executor(1);
        auto future = executor.submit(
            [release_future](const nlohmann::json&) mutable -> zoo::Expected<nlohmann::json> {
                release_future.wait_for(5s);
                return nlohmann::json{{"ok", true}};
            },
            nlohmann::json{});
        release->set_value();
        ASSERT_EQ(future.wait_for(3s), std::future_status::ready);
        ASSERT_TRUE(future.get().has_value());
    }
}

TEST(ToolExecutorTest, DestructorDoesNotBlockOnSlowHandler) {
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    {
        zoo::internal::agent::ToolExecutor executor(1);
        (void)executor.submit(
            [entered, release_future](const nlohmann::json&) mutable -> zoo::Expected<nlohmann::json> {
                entered->set_value();
                release_future.wait_for(5s);
                return nlohmann::json{{"ok", true}};
            },
            nlohmann::json{});
        ASSERT_EQ(entered_future.wait_for(3s), std::future_status::ready);
    }

    release->set_value();
}

TEST(ToolExecutorTest, HandlerExceptionMapsToToolExecutionFailed) {
    zoo::internal::agent::ToolExecutor executor(1);
    auto future = executor.submit(
        [](const nlohmann::json&) -> zoo::Expected<nlohmann::json> {
            throw std::runtime_error("boom");
        },
        nlohmann::json{});
    ASSERT_EQ(future.wait_for(3s), std::future_status::ready);
    const auto result = future.get();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolExecutionFailed);
}

TEST(ToolExecutorTest, ReusesWorkersAcrossManySubmissions) {
    zoo::internal::agent::ToolExecutor executor(2);
    std::mutex mutex;
    std::vector<std::thread::id> observed_threads;

    for (int i = 0; i < 8; ++i) {
        auto future = executor.submit(
            [&mutex, &observed_threads](const nlohmann::json&) -> zoo::Expected<nlohmann::json> {
                {
                    std::lock_guard lock(mutex);
                    observed_threads.push_back(std::this_thread::get_id());
                }
                return nlohmann::json{{"ok", true}};
            },
            nlohmann::json{});
        ASSERT_EQ(future.wait_for(3s), std::future_status::ready);
        ASSERT_TRUE(future.get().has_value());
    }

    std::sort(observed_threads.begin(), observed_threads.end());
    observed_threads.erase(std::unique(observed_threads.begin(), observed_threads.end()),
                           observed_threads.end());
    EXPECT_LE(observed_threads.size(), 2u);
}
