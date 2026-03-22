/**
 * @file test_token_accounting.cpp
 * @brief Unit tests for token accounting in Model history management.
 */

#include "zoo/core/types.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#include "zoo/core/model.hpp"
#undef private
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#include "../../extern/llama.cpp/common/chat.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace zoo::core {

struct Model::ToolCallingState {
    std::vector<common_chat_tool> tools;
    common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string> preserved_tokens;
    std::vector<std::string> additional_stops;
    bool thinking_forced_open = false;
    common_peg_arena parser;
};

} // namespace zoo::core

namespace {

zoo::ModelConfig make_config() {
    zoo::ModelConfig config;
    config.model_path = "unused.gguf";
    return config;
}

TEST(TokenAccountingTest, PlainMessageAccounting) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    zoo::Message msg = zoo::Message::assistant("hello");
    int via_estimate_message = model.estimate_message_tokens(msg);
    int via_estimate_tokens =
        model.estimate_tokens(msg.content) + zoo::core::Model::kTemplateOverheadPerMessage;

    EXPECT_EQ(via_estimate_message, via_estimate_tokens);
}

TEST(TokenAccountingTest, ToolCallMessageLargerThanPlain) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    std::vector<zoo::ToolCallInfo> calls = {{"id1", "add", R"({"a":1})"}};
    zoo::Message with_tool_calls = zoo::Message::assistant_with_tool_calls("", std::move(calls));
    zoo::Message plain = zoo::Message::assistant("");

    int tool_call_estimate = model.estimate_message_tokens(with_tool_calls);
    int plain_estimate = model.estimate_message_tokens(plain);

    EXPECT_GT(tool_call_estimate, plain_estimate);
}

TEST(TokenAccountingTest, ToolCallIdIncluded) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    zoo::Message long_id = zoo::Message::tool("result", "call_123");
    zoo::Message short_id = zoo::Message::tool("result", "x");

    int long_id_estimate = model.estimate_message_tokens(long_id);
    int short_id_estimate = model.estimate_message_tokens(short_id);

    EXPECT_GT(long_id_estimate, short_id_estimate);
}

TEST(TokenAccountingTest, AddMessageUpdatesEstimate) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    int before_user = model.estimated_tokens_;
    auto result = model.add_message(zoo::Message::user("hi").view());
    ASSERT_TRUE(result.has_value());

    int after_user = model.estimated_tokens_;
    zoo::Message user_msg = zoo::Message::user("hi");
    EXPECT_EQ(after_user - before_user, model.estimate_message_tokens(user_msg));

    std::vector<zoo::ToolCallInfo> calls = {{"tc1", "lookup", R"({"query":"test"})"}};
    zoo::Message tool_call_msg = zoo::Message::assistant_with_tool_calls("", std::move(calls));

    int before_tool = model.estimated_tokens_;
    auto result2 = model.add_message(tool_call_msg.view());
    ASSERT_TRUE(result2.has_value());

    int after_tool = model.estimated_tokens_;
    EXPECT_EQ(after_tool - before_tool, model.estimate_message_tokens(tool_call_msg));
}

TEST(TokenAccountingTest, RollbackRemovesToolCallCost) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    auto add_user = model.add_message(zoo::Message::user("hello").view());
    ASSERT_TRUE(add_user.has_value());

    std::vector<zoo::ToolCallInfo> calls = {{"id42", "search", R"({"q":"foo"})"}};
    zoo::Message tool_call_msg =
        zoo::Message::assistant_with_tool_calls("thinking", std::move(calls));

    auto add_assistant = model.add_message(tool_call_msg.view());
    ASSERT_TRUE(add_assistant.has_value());

    int tokens_after_add = model.estimated_tokens_;
    model.rollback_last_message();

    int tokens_after_rollback = model.estimated_tokens_;
    EXPECT_EQ(tokens_after_rollback,
              tokens_after_add - model.estimate_message_tokens(tool_call_msg));
}

TEST(TokenAccountingTest, ReplaceHistoryIncludesToolCalls) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    std::vector<zoo::ToolCallInfo> calls = {{"cid", "fn", R"({"x":1})"}};
    std::vector<zoo::Message> messages = {
        zoo::Message::user("ping"),
        zoo::Message::assistant_with_tool_calls("", std::move(calls)),
        zoo::Message::tool("result", "cid"),
    };

    int expected = 0;
    for (const auto& m : messages) {
        expected += model.estimate_message_tokens(m);
    }

    model.replace_history(zoo::HistorySnapshot{messages});

    EXPECT_EQ(model.estimated_tokens_, expected);
}

} // namespace
