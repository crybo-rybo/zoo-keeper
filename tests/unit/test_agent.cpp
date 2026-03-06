#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "mocks/mock_backend.hpp"
#include "zoo/agent.hpp"
#include "fixtures/tool_definitions.hpp"
#include "fixtures/sample_responses.hpp"

using namespace zoo::testing::tools;
using namespace zoo::testing::responses;

class AgentTest : public ::testing::Test {
protected:
    std::unique_ptr<zoo::Agent> make_agent(
        std::unique_ptr<zoo::testing::MockBackend> mock = nullptr
    ) {
        zoo::Config config;
        config.model_path = "/path/to/model.gguf";
        config.max_tokens = 512;

        if (!mock) mock = std::make_unique<zoo::testing::MockBackend>();
        auto result = zoo::Agent::create(config, std::move(mock));
        EXPECT_TRUE(result.has_value());
        return std::move(*result);
    }
};

TEST_F(AgentTest, CreateSuccess) {
    auto agent = make_agent();
    EXPECT_NE(agent, nullptr);
    EXPECT_TRUE(agent->is_running());
}

TEST_F(AgentTest, CreateFailsOnInvalidConfig) {
    zoo::Config config; // empty model_path
    auto result = zoo::Agent::create(config);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelPath);
}

TEST_F(AgentTest, CreateFailsOnBackendInit) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    auto mock = std::make_unique<zoo::testing::MockBackend>();
    mock->should_fail_initialize = true;
    auto result = zoo::Agent::create(config, std::move(mock));
    EXPECT_FALSE(result.has_value());
}

TEST_F(AgentTest, ChatBasic) {
    auto mock = std::make_unique<zoo::testing::MockBackend>();
    mock->default_response = "Hello back!";
    auto agent = make_agent(std::move(mock));

    auto handle = agent->chat(zoo::Message::user("Hello"));
    auto response = handle.future.get();
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(response->text, "Hello back!");
}

TEST_F(AgentTest, ChatUpdatesHistory) {
    auto agent = make_agent();
    auto handle = agent->chat(zoo::Message::user("Hello"));
    handle.future.get();

    auto history = agent->get_history();
    EXPECT_GE(history.size(), 2u);
}

TEST_F(AgentTest, SystemPrompt) {
    auto agent = make_agent();
    agent->set_system_prompt("You are helpful.");

    auto history = agent->get_history();
    ASSERT_GE(history.size(), 1u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[0].content, "You are helpful.");
}

TEST_F(AgentTest, ClearHistory) {
    auto agent = make_agent();
    auto handle = agent->chat(zoo::Message::user("Hello"));
    handle.future.get();
    EXPECT_FALSE(agent->get_history().empty());

    agent->clear_history();
    EXPECT_TRUE(agent->get_history().empty());
}

TEST_F(AgentTest, StopAndRejectNewRequests) {
    auto agent = make_agent();
    agent->stop();
    EXPECT_FALSE(agent->is_running());

    auto handle = agent->chat(zoo::Message::user("Hello"));
    auto response = handle.future.get();
    EXPECT_FALSE(response.has_value());
    EXPECT_EQ(response.error().code, zoo::ErrorCode::AgentNotRunning);
}

TEST_F(AgentTest, CancelRequest) {
    auto mock = std::make_unique<zoo::testing::MockBackend>();
    mock->generation_delay_ms = 500;
    auto agent = make_agent(std::move(mock));

    auto handle = agent->chat(zoo::Message::user("Hello"));
    agent->cancel(handle.id);
    auto response = handle.future.get();
    // May or may not be cancelled depending on timing
    (void)response;
}

TEST_F(AgentTest, RegisterTool) {
    auto agent = make_agent();
    auto result = agent->register_tool("add", "Add two numbers", {"a", "b"}, add);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(agent->tool_count(), 1);
}

TEST_F(AgentTest, ToolCallExecution) {
    auto mock = std::make_unique<zoo::testing::MockBackend>();
    mock->enqueue_response(TOOL_CALL_STEP1);
    mock->enqueue_response(FINAL_ANSWER_STEP2);
    auto agent = make_agent(std::move(mock));

    agent->register_tool("get_time", "Get time", {}, get_time);

    auto handle = agent->chat(zoo::Message::user("What time is it?"));
    auto response = handle.future.get();
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(response->text, FINAL_ANSWER_STEP2);
    EXPECT_FALSE(response->tool_calls.empty());
}

TEST_F(AgentTest, MultipleRequests) {
    auto agent = make_agent();

    auto h1 = agent->chat(zoo::Message::user("First"));
    auto r1 = h1.future.get();
    EXPECT_TRUE(r1.has_value());

    auto h2 = agent->chat(zoo::Message::user("Second"));
    auto r2 = h2.future.get();
    EXPECT_TRUE(r2.has_value());
}

TEST_F(AgentTest, UniqueRequestIds) {
    auto agent = make_agent();
    auto h1 = agent->chat(zoo::Message::user("First"));
    auto h2 = agent->chat(zoo::Message::user("Second"));
    EXPECT_NE(h1.id, h2.id);
    h1.future.get();
    h2.future.get();
}

TEST_F(AgentTest, StreamingCallback) {
    auto mock = std::make_unique<zoo::testing::MockBackend>();
    mock->default_response = "word1 word2 word3";
    auto agent = make_agent(std::move(mock));

    std::atomic<int> token_count{0};
    auto handle = agent->chat(zoo::Message::user("Hello"), [&token_count](std::string_view) {
        token_count.fetch_add(1, std::memory_order_relaxed);
    });
    auto response = handle.future.get();
    EXPECT_TRUE(response.has_value());
    EXPECT_GT(token_count.load(), 0);
}
