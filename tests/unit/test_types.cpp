#include <gtest/gtest.h>
#include "zoo/core/types.hpp"

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

TEST(ConfigTest, Equality) {
    zoo::Config c1, c2;
    c1.model_path = "/path/to/model.gguf";
    c2.model_path = "/path/to/model.gguf";
    EXPECT_EQ(c1, c2);
    c2.context_size = 4096;
    EXPECT_NE(c1, c2);
}

TEST(TokenUsageTest, Defaults) {
    zoo::TokenUsage usage;
    EXPECT_EQ(usage.prompt_tokens, 0);
    EXPECT_EQ(usage.completion_tokens, 0);
    EXPECT_EQ(usage.total_tokens, 0);
}

TEST(ResponseTest, Defaults) {
    zoo::Response response;
    EXPECT_TRUE(response.text.empty());
    EXPECT_TRUE(response.tool_calls.empty());
}
