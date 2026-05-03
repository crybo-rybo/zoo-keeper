/**
 * @file test_model_tool_calling.cpp
 * @brief Unit tests for internal Model tool-calling state refresh.
 */

#include "core/model_test_access.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

using zoo::core::ModelTestAccess;

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
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    auto templates = common_chat_templates_init(nullptr, peg_native_tool_template());
    ASSERT_TRUE(templates);
    ModelTestAccess::chat_templates(*model).reset(templates.release());

    auto state = std::make_unique<ModelTestAccess::ToolCallingState>();
    state->tools.push_back(
        {"echo", "Echo text",
         R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})"});
    state->parser_params.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    state->grammar = "stale-grammar";
    state->grammar_lazy = false;
    state->grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "stale-trigger"});
    state->preserved_tokens = {"stale-token"};
    state->additional_stops = {"stale-stop"};

    ModelTestAccess::tool_state(*model) = std::move(state);
    ModelTestAccess::set_sampler_policy(
        *model, ModelTestAccess::SamplerPolicy::native_tool_call("stale-grammar"));
    ModelTestAccess::messages(*model).push_back(zoo::Message::user("hello"));

    auto prompt = ModelTestAccess::render_prompt_delta(*model);
    ASSERT_TRUE(prompt.has_value()) << prompt.error().to_string();
    EXPECT_FALSE(prompt->empty());

    ASSERT_NE(ModelTestAccess::tool_state(*model), nullptr);
    EXPECT_EQ(ModelTestAccess::tool_state(*model)->parser_params.format,
              COMMON_CHAT_FORMAT_PEG_NATIVE);
    EXPECT_FALSE(ModelTestAccess::tool_state(*model)->grammar.empty());
    EXPECT_EQ(ModelTestAccess::sampler_policy(*model).grammar,
              ModelTestAccess::tool_state(*model)->grammar);
    EXPECT_EQ(ModelTestAccess::sampler_policy(*model).mode,
              ModelTestAccess::GrammarMode::NativeToolCall);
    EXPECT_TRUE(ModelTestAccess::tool_state(*model)->grammar_lazy);
    ASSERT_EQ(ModelTestAccess::tool_state(*model)->grammar_triggers.size(), 1u);
    EXPECT_EQ(ModelTestAccess::tool_state(*model)->grammar_triggers.front().type,
              COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
    EXPECT_EQ(ModelTestAccess::tool_state(*model)->grammar_triggers.front().value, "[TOOL_CALLS]");
    EXPECT_FALSE(ModelTestAccess::tool_state(*model)->preserved_tokens.empty());
    EXPECT_TRUE(ModelTestAccess::tool_state(*model)->additional_stops.empty());
    EXPECT_EQ(ModelTestAccess::tool_state(*model)->parser_params.generation_prompt, "assistant:");
    EXPECT_FALSE(ModelTestAccess::tool_state(*model)->parser_params.parser.empty());
}

TEST(ModelToolCallingTest, ParseToolResponseExtractsStructuredCalls) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    auto templates = common_chat_templates_init(nullptr, peg_native_tool_template());
    ASSERT_TRUE(templates);
    ModelTestAccess::chat_templates(*model).reset(templates.release());

    auto state = std::make_unique<ModelTestAccess::ToolCallingState>();
    state->parser_params.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    state->tools.push_back(
        {"echo", "Echo text",
         R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})"});

    ModelTestAccess::tool_state(*model) = std::move(state);
    ModelTestAccess::set_sampler_policy(
        *model, ModelTestAccess::SamplerPolicy::native_tool_call("stale-grammar"));
    ModelTestAccess::messages(*model).push_back(zoo::Message::user("hello"));

    auto prompt = ModelTestAccess::render_prompt_delta(*model);
    ASSERT_TRUE(prompt.has_value()) << prompt.error().to_string();
    ASSERT_EQ(ModelTestAccess::tool_state(*model)->parser_params.format,
              COMMON_CHAT_FORMAT_PEG_NATIVE);
    ASSERT_EQ(ModelTestAccess::tool_state(*model)->parser_params.generation_prompt, "assistant:");

    std::string tool_text = R"([TOOL_CALLS]echo[ARGS]{"text":"hi"})";
    auto parsed = model->parse_tool_response(tool_text);

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
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    EXPECT_EQ(ModelTestAccess::sampler_policy(*model).mode, ModelTestAccess::GrammarMode::Plain);
    EXPECT_EQ(ModelTestAccess::tool_state(*model), nullptr);

    std::string generated_text = "Hello, world!";
    zoo::Message msg = (ModelTestAccess::sampler_policy(*model).is_native_tool_call() &&
                        ModelTestAccess::tool_state(*model))
                           ? zoo::Message::assistant_with_tool_calls("", {})
                           : zoo::Message::assistant(generated_text);

    EXPECT_EQ(msg.role, zoo::Role::Assistant);
    EXPECT_EQ(msg.content, "Hello, world!");
    EXPECT_TRUE(msg.tool_calls.empty());
}

TEST(ModelToolCallingTest, RenderPromptDeltaDoesNotOverwriteSchemaPolicy) {
    auto model = ModelTestAccess::make(make_config(), zoo::GenerationOptions{});

    auto templates = common_chat_templates_init(nullptr, peg_native_tool_template());
    ASSERT_TRUE(templates);
    ModelTestAccess::chat_templates(*model).reset(templates.release());

    auto state = std::make_unique<ModelTestAccess::ToolCallingState>();
    state->tools.push_back(
        {"echo", "Echo text",
         R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]})"});
    state->grammar = "tool-grammar";
    state->additional_stops = {"tool-stop"};

    ModelTestAccess::tool_state(*model) = std::move(state);
    ModelTestAccess::set_sampler_policy(*model,
                                        ModelTestAccess::SamplerPolicy::schema("schema-grammar"));
    ModelTestAccess::messages(*model).push_back(zoo::Message::user("extract"));

    auto prompt = ModelTestAccess::render_prompt_delta(*model);
    ASSERT_TRUE(prompt.has_value()) << prompt.error().to_string();

    EXPECT_EQ(ModelTestAccess::sampler_policy(*model).mode, ModelTestAccess::GrammarMode::Schema);
    EXPECT_EQ(ModelTestAccess::sampler_policy(*model).grammar, "schema-grammar");
    EXPECT_EQ(ModelTestAccess::tool_state(*model)->grammar, "tool-grammar");
}

} // namespace
