/**
 * @file test_token_accounting.cpp
 * @brief Unit tests for token accounting in Model history management.
 */

#include "core/model_test_access.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace {

using zoo::core::ModelTestAccess;

zoo::ModelConfig make_config() {
    zoo::ModelConfig config;
    config.model_path = "unused.gguf";
    return config;
}

int estimate_messages(zoo::core::Model& model, const std::vector<zoo::Message>& messages) {
    int expected = 0;
    for (const auto& message : messages) {
        expected += ModelTestAccess::estimate_message_tokens(model, message);
    }
    return expected;
}

TEST(TokenAccountingTest, PlainMessageAccounting) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    zoo::Message msg = zoo::Message::assistant("hello");
    int via_estimate_message = ModelTestAccess::estimate_message_tokens(*model, msg);
    int via_estimate_tokens = 1 + 8;

    EXPECT_EQ(via_estimate_message, via_estimate_tokens);
}

TEST(TokenAccountingTest, ToolCallMessageLargerThanPlain) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    std::vector<zoo::ToolCallInfo> calls = {{"id1", "add", R"({"a":1})"}};
    zoo::Message with_tool_calls = zoo::Message::assistant_with_tool_calls("", std::move(calls));
    zoo::Message plain = zoo::Message::assistant("");

    int tool_call_estimate = ModelTestAccess::estimate_message_tokens(*model, with_tool_calls);
    int plain_estimate = ModelTestAccess::estimate_message_tokens(*model, plain);

    EXPECT_GT(tool_call_estimate, plain_estimate);
}

TEST(TokenAccountingTest, ToolCallIdIncluded) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    zoo::Message long_id = zoo::Message::tool("result", "call_123");
    zoo::Message short_id = zoo::Message::tool("result", "x");

    int long_id_estimate = ModelTestAccess::estimate_message_tokens(*model, long_id);
    int short_id_estimate = ModelTestAccess::estimate_message_tokens(*model, short_id);

    EXPECT_GT(long_id_estimate, short_id_estimate);
}

TEST(TokenAccountingTest, AddMessageUpdatesEstimate) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    int before_user = model->estimated_tokens();
    auto result = model->add_message(zoo::Message::user("hi").view());
    ASSERT_TRUE(result.has_value());

    int after_user = model->estimated_tokens();
    zoo::Message user_msg = zoo::Message::user("hi");
    EXPECT_EQ(after_user - before_user, ModelTestAccess::estimate_message_tokens(*model, user_msg));

    std::vector<zoo::ToolCallInfo> calls = {{"tc1", "lookup", R"({"query":"test"})"}};
    zoo::Message tool_call_msg = zoo::Message::assistant_with_tool_calls("", std::move(calls));

    int before_tool = model->estimated_tokens();
    auto result2 = model->add_message(tool_call_msg.view());
    ASSERT_TRUE(result2.has_value());

    int after_tool = model->estimated_tokens();
    EXPECT_EQ(after_tool - before_tool,
              ModelTestAccess::estimate_message_tokens(*model, tool_call_msg));
}

TEST(TokenAccountingTest, RollbackRemovesToolCallCost) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    auto add_user = model->add_message(zoo::Message::user("hello").view());
    ASSERT_TRUE(add_user.has_value());

    std::vector<zoo::ToolCallInfo> calls = {{"id42", "search", R"({"q":"foo"})"}};
    zoo::Message tool_call_msg =
        zoo::Message::assistant_with_tool_calls("thinking", std::move(calls));

    auto add_assistant = model->add_message(tool_call_msg.view());
    ASSERT_TRUE(add_assistant.has_value());

    int tokens_after_add = model->estimated_tokens();
    ModelTestAccess::rollback_last_message(*model);

    int tokens_after_rollback = model->estimated_tokens();
    EXPECT_EQ(tokens_after_rollback,
              tokens_after_add - ModelTestAccess::estimate_message_tokens(*model, tool_call_msg));
}

TEST(TokenAccountingTest, ReplaceHistoryIncludesToolCalls) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    std::vector<zoo::ToolCallInfo> calls = {{"cid", "fn", R"({"x":1})"}};
    std::vector<zoo::Message> messages = {
        zoo::Message::user("ping"),
        zoo::Message::assistant_with_tool_calls("", std::move(calls)),
        zoo::Message::tool("result", "cid"),
    };

    int expected = 0;
    for (const auto& m : messages) {
        expected += ModelTestAccess::estimate_message_tokens(*model, m);
    }

    model->replace_history(zoo::HistorySnapshot{messages});

    EXPECT_EQ(model->estimated_tokens(), expected);
}

TEST(TokenAccountingTest, TrimHistoryKeepsSystemPromptAndLatestExchange) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});
    std::vector<zoo::Message> messages = {
        zoo::Message::system("system"),
        zoo::Message::user("old question"),
        zoo::Message::assistant("old answer"),
        zoo::Message::user("new question"),
        zoo::Message::assistant("new answer"),
    };
    model->replace_history(zoo::HistorySnapshot{messages});

    model->trim_history(2);

    const auto history = model->get_history();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[1].content, "new question");
    EXPECT_EQ(history[2].content, "new answer");
    EXPECT_EQ(model->estimated_tokens(), estimate_messages(*model, history.messages));
}

TEST(TokenAccountingTest, TrimHistoryStartsAtUserBoundary) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});
    std::vector<zoo::ToolCallInfo> calls = {{"call_1", "lookup", R"({"q":"old"})"}};
    std::vector<zoo::Message> messages = {
        zoo::Message::system("system"),
        zoo::Message::user("old question"),
        zoo::Message::assistant_with_tool_calls("old tool call", std::move(calls)),
        zoo::Message::tool("old result", "call_1"),
        zoo::Message::user("new question"),
        zoo::Message::assistant("new answer"),
    };
    model->replace_history(zoo::HistorySnapshot{messages});

    model->trim_history(3);

    const auto history = model->get_history();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].role, zoo::Role::System);
    EXPECT_EQ(history[1].role, zoo::Role::User);
    EXPECT_EQ(history[1].content, "new question");
    EXPECT_EQ(history[2].content, "new answer");
    EXPECT_EQ(model->estimated_tokens(), estimate_messages(*model, history.messages));
}

} // namespace
