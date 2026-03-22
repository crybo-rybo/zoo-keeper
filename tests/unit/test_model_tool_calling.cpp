/**
 * @file test_model_tool_calling.cpp
 * @brief Unit tests for internal Model tool-calling state refresh.
 */

#include "zoo/core/types.hpp"
#include "zoo/internal/core/stream_filter.hpp"

#include <gtest/gtest.h>

#include <memory>
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
    ToolCallTriggerMatcher trigger_matcher;
    std::vector<std::string> preserved_tokens;
    std::vector<std::string> additional_stops;
    bool thinking_forced_open = false;
    common_peg_arena parser;
};

} // namespace zoo::core

namespace {

std::string peg_native_tool_template() {
    return R"JINJA(
{# [SYSTEM_PROMPT] [TOOL_CALLS] [ARGS] #}
{% for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}assistant:{% endif %}
)JINJA";
}

zoo::ModelConfig make_config() {
    zoo::ModelConfig config;
    config.model_path = "unused.gguf";
    return config;
}

TEST(ModelToolCallingTest, RenderPromptDeltaRefreshesParserAndGrammarState) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    auto templates = common_chat_templates_init(nullptr, peg_native_tool_template());
    ASSERT_TRUE(templates);
    model.chat_templates_.reset(templates.release());

    auto state = std::make_unique<zoo::core::Model::ToolCallingState>();
    state->tools.push_back(
        {"echo", "Echo text",
         R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})"});
    state->format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    state->grammar = "stale-grammar";
    state->grammar_lazy = false;
    state->grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "stale-trigger"});
    state->preserved_tokens = {"stale-token"};
    state->additional_stops = {"stale-stop"};
    state->thinking_forced_open = true;

    model.tool_grammar_str_ = state->grammar;
    model.tool_state_ = std::move(state);
    model.grammar_mode_ = zoo::core::Model::GrammarMode::NativeToolCall;
    model.messages_.push_back(zoo::Message::user("hello"));

    auto prompt = model.render_prompt_delta();
    ASSERT_TRUE(prompt.has_value()) << prompt.error().to_string();
    EXPECT_FALSE(prompt->empty());

    ASSERT_NE(model.tool_state_, nullptr);
    EXPECT_EQ(model.tool_state_->format, COMMON_CHAT_FORMAT_PEG_NATIVE);
    EXPECT_FALSE(model.tool_state_->grammar.empty());
    EXPECT_EQ(model.tool_grammar_str_, model.tool_state_->grammar);
    EXPECT_TRUE(model.tool_state_->grammar_lazy);
    ASSERT_EQ(model.tool_state_->grammar_triggers.size(), 1u);
    EXPECT_EQ(model.tool_state_->grammar_triggers.front().type, COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
    EXPECT_EQ(model.tool_state_->grammar_triggers.front().value, "[TOOL_CALLS]");
    EXPECT_FALSE(model.tool_state_->preserved_tokens.empty());
    EXPECT_TRUE(model.tool_state_->additional_stops.empty());
    EXPECT_FALSE(model.tool_state_->thinking_forced_open);
    EXPECT_FALSE(model.tool_state_->parser.empty());
}

TEST(ModelToolCallingTest, ParseToolResponseExtractsStructuredCalls) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    auto templates = common_chat_templates_init(nullptr, peg_native_tool_template());
    ASSERT_TRUE(templates);
    model.chat_templates_.reset(templates.release());

    auto state = std::make_unique<zoo::core::Model::ToolCallingState>();
    state->format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    state->tools.push_back(
        {"echo", "Echo text",
         R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})"});

    model.tool_state_ = std::move(state);
    model.grammar_mode_ = zoo::core::Model::GrammarMode::NativeToolCall;
    model.messages_.push_back(zoo::Message::user("hello"));

    // Render to populate the parser state (updates tool_state_->format and parser).
    auto prompt = model.render_prompt_delta();
    ASSERT_TRUE(prompt.has_value()) << prompt.error().to_string();
    ASSERT_EQ(model.tool_state_->format, COMMON_CHAT_FORMAT_PEG_NATIVE);

    // PEG_NATIVE format: [TOOL_CALLS]<name>[ARGS]<json-args>
    std::string tool_text = R"([TOOL_CALLS]echo[ARGS]{"text":"hi"})";
    auto parsed = model.parse_tool_response(tool_text);

    ASSERT_FALSE(parsed.tool_calls.empty());
    EXPECT_EQ(parsed.tool_calls[0].name, "echo");
}

TEST(ModelToolCallingTest, AssistantWithToolCallsPreservesStructure) {
    std::vector<zoo::ToolCallInfo> calls = {{"call_1", "echo", R"({"text":"hi"})"}};
    auto msg = zoo::Message::assistant_with_tool_calls("visible text", calls);

    EXPECT_EQ(msg.role, zoo::Role::Assistant);
    EXPECT_EQ(msg.content, "visible text");
    ASSERT_EQ(msg.tool_calls.size(), 1u);
    EXPECT_EQ(msg.tool_calls[0].name, "echo");
    EXPECT_EQ(msg.tool_calls[0].id, "call_1");
    EXPECT_EQ(msg.tool_calls[0].arguments_json, R"({"text":"hi"})");
}

TEST(ModelToolCallingTest, PlainAssistantMessageWhenNoToolState) {
    zoo::core::Model model(make_config(), zoo::GenerationOptions{});

    // No tool_state_ set, grammar_mode_ defaults to None.
    EXPECT_EQ(model.grammar_mode_, zoo::core::Model::GrammarMode::None);
    EXPECT_EQ(model.tool_state_, nullptr);

    // Simulate the generate() branching: no NativeToolCall mode → plain assistant message.
    std::string generated_text = "Hello, world!";
    zoo::Message msg =
        (model.grammar_mode_ == zoo::core::Model::GrammarMode::NativeToolCall && model.tool_state_)
            ? zoo::Message::assistant_with_tool_calls("", {})
            : zoo::Message::assistant(generated_text);

    EXPECT_EQ(msg.role, zoo::Role::Assistant);
    EXPECT_EQ(msg.content, "Hello, world!");
    EXPECT_TRUE(msg.tool_calls.empty());
}

} // namespace
