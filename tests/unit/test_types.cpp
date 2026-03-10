/**
 * @file test_types.cpp
 * @brief Unit tests for shared core value types and validation helpers.
 */

#include "zoo/core/types.hpp"
#include <gtest/gtest.h>

TEST(RoleTest, RoleToString) {
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::System), "system");
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::User), "user");
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::Assistant), "assistant");
    EXPECT_STREQ(zoo::role_to_string(zoo::Role::Tool), "tool");
}

TEST(MessageTest, FactoryMethods) {
    auto sys = zoo::Message::system("System message");
    EXPECT_EQ(sys.role, zoo::Role::System);
    EXPECT_EQ(sys.content, "System message");
    EXPECT_FALSE(sys.tool_call_id.has_value());

    auto user = zoo::Message::user("User message");
    EXPECT_EQ(user.role, zoo::Role::User);

    auto assistant = zoo::Message::assistant("Assistant message");
    EXPECT_EQ(assistant.role, zoo::Role::Assistant);

    auto tool = zoo::Message::tool("Tool result", "call_123");
    EXPECT_EQ(tool.role, zoo::Role::Tool);
    ASSERT_TRUE(tool.tool_call_id.has_value());
    EXPECT_EQ(*tool.tool_call_id, "call_123");
}

TEST(MessageTest, Equality) {
    auto msg1 = zoo::Message::user("Hello");
    auto msg2 = zoo::Message::user("Hello");
    auto msg3 = zoo::Message::user("World");
    auto msg4 = zoo::Message::assistant("Hello");

    EXPECT_EQ(msg1, msg2);
    EXPECT_NE(msg1, msg3);
    EXPECT_NE(msg1, msg4);
}

TEST(ErrorTest, Construction) {
    zoo::Error err(zoo::ErrorCode::InvalidConfig, "Test error");
    EXPECT_EQ(err.code, zoo::ErrorCode::InvalidConfig);
    EXPECT_EQ(err.message, "Test error");
    EXPECT_FALSE(err.context.has_value());
}

TEST(ErrorTest, ToString) {
    zoo::Error err(zoo::ErrorCode::InvalidConfig, "Configuration is invalid");
    std::string str = err.to_string();
    EXPECT_NE(str.find("100"), std::string::npos);
    EXPECT_NE(str.find("Configuration is invalid"), std::string::npos);
}

TEST(ErrorTest, ToStringWithContext) {
    zoo::Error err(zoo::ErrorCode::InferenceFailed, "Inference failed", "Out of memory");
    std::string str = err.to_string();
    EXPECT_NE(str.find("Out of memory"), std::string::npos);
}

TEST(ErrorTest, Expected) {
    zoo::Expected<int> success = 42;
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(*success, 42);

    zoo::Expected<int> failure = std::unexpected(zoo::Error{zoo::ErrorCode::Unknown, "Failed"});
    EXPECT_FALSE(failure.has_value());
    EXPECT_EQ(failure.error().code, zoo::ErrorCode::Unknown);
}

TEST(SamplingParamsTest, Defaults) {
    zoo::SamplingParams params;
    EXPECT_FLOAT_EQ(params.temperature, 0.7f);
    EXPECT_FLOAT_EQ(params.top_p, 0.9f);
    EXPECT_EQ(params.top_k, 40);
}

TEST(SamplingParamsTest, Equality) {
    zoo::SamplingParams p1, p2;
    EXPECT_EQ(p1, p2);
    p2.temperature = 0.5f;
    EXPECT_NE(p1, p2);
}

TEST(SamplingParamsTest, ValidateDefaults) {
    zoo::SamplingParams params;
    EXPECT_TRUE(params.validate().has_value());
}

TEST(SamplingParamsTest, ValidateNegativeTemperature) {
    zoo::SamplingParams params;
    params.temperature = -0.1f;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(SamplingParamsTest, ValidateZeroTemperature) {
    zoo::SamplingParams params;
    params.temperature = 0.0f;
    EXPECT_TRUE(params.validate().has_value());
}

TEST(SamplingParamsTest, ValidateTopPBelowZero) {
    zoo::SamplingParams params;
    params.top_p = -0.1f;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(SamplingParamsTest, ValidateTopPAboveOne) {
    zoo::SamplingParams params;
    params.top_p = 1.1f;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(SamplingParamsTest, ValidateTopPBoundary) {
    zoo::SamplingParams params;
    params.top_p = 0.0f;
    EXPECT_TRUE(params.validate().has_value());
    params.top_p = 1.0f;
    EXPECT_TRUE(params.validate().has_value());
}

TEST(SamplingParamsTest, ValidateTopKZero) {
    zoo::SamplingParams params;
    params.top_k = 0;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(SamplingParamsTest, ValidateTopKNegative) {
    zoo::SamplingParams params;
    params.top_k = -1;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(SamplingParamsTest, ValidateNegativeRepeatPenalty) {
    zoo::SamplingParams params;
    params.repeat_penalty = -1.0f;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(SamplingParamsTest, ValidateNegativeRepeatLastN) {
    zoo::SamplingParams params;
    params.repeat_last_n = -1;
    auto result = params.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(ConfigTest, ValidationSuccess) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    EXPECT_TRUE(config.validate().has_value());
}

TEST(ConfigTest, ValidationEmptyModelPath) {
    zoo::Config config;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidModelPath);
}

TEST(ConfigTest, ValidationInvalidContextSize) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.context_size = 0;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidContextSize);
}

TEST(ConfigTest, ValidationInvalidMaxTokens) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.max_tokens = 0;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ConfigTest, ValidationRejectsBadSampling) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.sampling.temperature = -1.0f;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidSamplingParams);
}

TEST(ConfigTest, ValidationRejectsZeroToolIterations) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.max_tool_iterations = 0;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ConfigTest, ValidationRejectsNegativeToolRetries) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.max_tool_retries = -1;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ConfigTest, ValidationRejectsZeroHistoryBudget) {
    zoo::Config config;
    config.model_path = "/path/to/model.gguf";
    config.max_history_messages = 0;
    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidConfig);
}

TEST(ConfigTest, DefaultQueueCapacity) {
    zoo::Config config;
    EXPECT_EQ(config.request_queue_capacity, 64u);
}

TEST(ConfigTest, DefaultGpuOffloadIsDisabled) {
    zoo::Config config;
    EXPECT_EQ(config.n_gpu_layers, 0);
}

TEST(ConfigTest, DefaultHistoryBudgetIsBounded) {
    zoo::Config config;
    EXPECT_EQ(config.max_history_messages, 64u);
}

TEST(ConfigTest, DefaultToolLimits) {
    zoo::Config config;
    EXPECT_EQ(config.max_tool_iterations, 5);
    EXPECT_EQ(config.max_tool_retries, 2);
}

TEST(ConfigTest, Equality) {
    zoo::Config c1, c2;
    c1.model_path = "/path/to/model.gguf";
    c2.model_path = "/path/to/model.gguf";
    EXPECT_EQ(c1, c2);
    c2.context_size = 4096;
    EXPECT_NE(c1, c2);
}

TEST(ConfigTest, EqualityToolLimits) {
    zoo::Config c1, c2;
    c1.model_path = "/path/to/model.gguf";
    c2.model_path = "/path/to/model.gguf";
    EXPECT_EQ(c1, c2);
    c2.max_tool_iterations = 10;
    EXPECT_NE(c1, c2);
}

TEST(ConfigTest, EqualityHistoryBudget) {
    zoo::Config c1, c2;
    c1.model_path = "/path/to/model.gguf";
    c2.model_path = "/path/to/model.gguf";
    EXPECT_EQ(c1, c2);
    c2.max_history_messages = 32;
    EXPECT_NE(c1, c2);
}

TEST(RoleValidationTest, EmptyHistoryAcceptsUser) {
    std::vector<zoo::Message> history;
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::User).has_value());
}

TEST(RoleValidationTest, EmptyHistoryAcceptsSystem) {
    std::vector<zoo::Message> history;
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::System).has_value());
}

TEST(RoleValidationTest, EmptyHistoryRejectsTool) {
    std::vector<zoo::Message> history;
    auto result = zoo::validate_role_sequence(history, zoo::Role::Tool);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST(RoleValidationTest, SystemOnlyAllowedAtBeginning) {
    std::vector<zoo::Message> history = {zoo::Message::user("Hello")};
    auto result = zoo::validate_role_sequence(history, zoo::Role::System);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST(RoleValidationTest, ConsecutiveSameRoleFails) {
    std::vector<zoo::Message> history = {zoo::Message::user("Hello")};
    auto result = zoo::validate_role_sequence(history, zoo::Role::User);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidMessageSequence);
}

TEST(RoleValidationTest, ConsecutiveToolAllowed) {
    std::vector<zoo::Message> history = {zoo::Message::user("Hello"),
                                         zoo::Message::assistant("I'll use tools"),
                                         zoo::Message::tool("result1", "id1")};
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::Tool).has_value());
}

TEST(RoleValidationTest, NormalAlternation) {
    std::vector<zoo::Message> history = {zoo::Message::user("Hello")};
    EXPECT_TRUE(zoo::validate_role_sequence(history, zoo::Role::Assistant).has_value());
}

TEST(TokenUsageTest, Defaults) {
    zoo::TokenUsage usage;
    EXPECT_EQ(usage.prompt_tokens, 0);
    EXPECT_EQ(usage.completion_tokens, 0);
    EXPECT_EQ(usage.total_tokens, 0);
}

TEST(ToolInvocationTest, Defaults) {
    zoo::ToolInvocation invocation;
    EXPECT_TRUE(invocation.id.empty());
    EXPECT_TRUE(invocation.name.empty());
    EXPECT_TRUE(invocation.arguments_json.empty());
    EXPECT_EQ(invocation.status, zoo::ToolInvocationStatus::Succeeded);
    EXPECT_FALSE(invocation.result_json.has_value());
    EXPECT_FALSE(invocation.error.has_value());
}

TEST(ResponseTest, Defaults) {
    zoo::Response response;
    EXPECT_TRUE(response.text.empty());
    EXPECT_TRUE(response.tool_invocations.empty());
}
