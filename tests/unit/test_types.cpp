/**
 * @file test_types.cpp
 * @brief Unit tests for shared core value types and validation helpers.
 */

#include "zoo/core/json.hpp"
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

TEST(SamplingParamsJsonTest, RoundTripsDefaultValues) {
    const zoo::SamplingParams params;
    const nlohmann::json json = params;

    EXPECT_EQ(json.at("temperature"), 0.7f);
    EXPECT_EQ(json.at("top_p"), 0.9f);
    EXPECT_EQ(json.at("top_k"), 40);
    EXPECT_EQ(json.at("repeat_penalty"), 1.1f);
    EXPECT_EQ(json.at("repeat_last_n"), 64);
    EXPECT_EQ(json.at("seed"), -1);

    const auto round_trip = json.get<zoo::SamplingParams>();
    EXPECT_EQ(round_trip, params);
}

TEST(SamplingParamsJsonTest, RejectsUnknownKeys) {
    const nlohmann::json json = {{"temperature", 0.7f}, {"unsupported", true}};
    EXPECT_THROW((void)json.get<zoo::SamplingParams>(), std::invalid_argument);
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

TEST(ConfigJsonTest, RoundTripsSerializableFields) {
    zoo::Config config;
    config.model_path = "/tmp/model.gguf";
    config.context_size = 4096;
    config.n_gpu_layers = 12;
    config.use_mmap = false;
    config.use_mlock = true;
    config.sampling.temperature = 0.2f;
    config.sampling.top_p = 0.8f;
    config.sampling.top_k = 12;
    config.sampling.repeat_penalty = 1.3f;
    config.sampling.repeat_last_n = 16;
    config.sampling.seed = 7;
    config.max_tokens = 256;
    config.stop_sequences = {"</tool_call>", "User:"};
    config.system_prompt = "You are concise.";
    config.max_history_messages = 8;
    config.request_queue_capacity = 4;
    config.max_tool_iterations = 3;
    config.max_tool_retries = 1;
    config.on_token = [](std::string_view) { return zoo::TokenAction::Continue; };

    const nlohmann::json json = config;
    EXPECT_FALSE(json.contains("on_token"));
    EXPECT_EQ(json.at("sampling").at("repeat_last_n"), 16);
    EXPECT_EQ(json.at("request_queue_capacity"), 4u);
    EXPECT_EQ(json.at("system_prompt"), "You are concise.");

    const auto round_trip = json.get<zoo::Config>();
    EXPECT_EQ(round_trip, config);
    EXPECT_FALSE(round_trip.on_token.has_value());
}

TEST(ConfigJsonTest, OmitsUnsetSystemPrompt) {
    zoo::Config config;
    config.model_path = "/tmp/model.gguf";

    const nlohmann::json json = config;
    EXPECT_FALSE(json.contains("system_prompt"));
}

TEST(ConfigJsonTest, AppliesDefaultsToOmittedFields) {
    const nlohmann::json json = {{"model_path", "/tmp/model.gguf"}};
    const auto config = json.get<zoo::Config>();

    EXPECT_EQ(config.model_path, "/tmp/model.gguf");
    EXPECT_EQ(config.context_size, 8192);
    EXPECT_EQ(config.max_tokens, -1);
    EXPECT_EQ(config.request_queue_capacity, 64u);
    EXPECT_FALSE(config.system_prompt.has_value());
}

TEST(ConfigJsonTest, RejectsMissingModelPath) {
    const nlohmann::json json = {{"context_size", 4096}};
    EXPECT_THROW((void)json.get<zoo::Config>(), std::invalid_argument);
}

TEST(ConfigJsonTest, RejectsUnknownTopLevelKeys) {
    const nlohmann::json json = {{"model_path", "/tmp/model.gguf"}, {"tools", true}};
    EXPECT_THROW((void)json.get<zoo::Config>(), std::invalid_argument);
}

TEST(ConfigJsonTest, RejectsUnknownSamplingKeys) {
    const nlohmann::json json = {{"model_path", "/tmp/model.gguf"},
                                 {"sampling", {{"temperature", 0.7f}, {"extra", 1}}}};
    EXPECT_THROW((void)json.get<zoo::Config>(), std::invalid_argument);
}

TEST(ConfigJsonTest, RejectsTypeMismatches) {
    const nlohmann::json json = {{"model_path", 42}};
    EXPECT_THROW((void)json.get<zoo::Config>(), std::exception);
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

TEST(ToolInvocationTest, StatusToString) {
    EXPECT_STREQ(zoo::to_string(zoo::ToolInvocationStatus::Succeeded), "succeeded");
    EXPECT_STREQ(zoo::to_string(zoo::ToolInvocationStatus::ValidationFailed), "validation_failed");
    EXPECT_STREQ(zoo::to_string(zoo::ToolInvocationStatus::ExecutionFailed), "execution_failed");
}

TEST(ToolInvocationTest, ConstructedFieldsMatch) {
    zoo::ToolInvocation invocation;
    invocation.id = "call_123";
    invocation.name = "add";
    invocation.arguments_json = R"({"a":1,"b":2})";
    invocation.status = zoo::ToolInvocationStatus::Succeeded;
    invocation.result_json = R"({"result":3})";

    EXPECT_EQ(invocation.id, "call_123");
    EXPECT_EQ(invocation.name, "add");
    EXPECT_TRUE(invocation.result_json.has_value());
    EXPECT_FALSE(invocation.error.has_value());
}

TEST(ToolInvocationTest, FailedInvocationCarriesError) {
    zoo::ToolInvocation invocation;
    invocation.id = "call_456";
    invocation.name = "greet";
    invocation.arguments_json = R"({"name":42})";
    invocation.status = zoo::ToolInvocationStatus::ValidationFailed;
    invocation.error = zoo::Error{zoo::ErrorCode::ToolValidationFailed, "wrong type"};

    EXPECT_EQ(invocation.status, zoo::ToolInvocationStatus::ValidationFailed);
    ASSERT_TRUE(invocation.error.has_value());
    EXPECT_EQ(invocation.error->code, zoo::ErrorCode::ToolValidationFailed);
    EXPECT_FALSE(invocation.result_json.has_value());
}

TEST(ResponseTest, Defaults) {
    zoo::Response response;
    EXPECT_TRUE(response.text.empty());
    EXPECT_TRUE(response.tool_invocations.empty());
}
