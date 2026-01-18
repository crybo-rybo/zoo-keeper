#include <gtest/gtest.h>
#include "zoo/types.hpp"

using namespace zoo;

// ============================================================================
// Message Tests
// ============================================================================

TEST(MessageTest, FactoryMethods) {
    auto sys = Message::system("System message");
    EXPECT_EQ(sys.role, Role::System);
    EXPECT_EQ(sys.content, "System message");
    EXPECT_FALSE(sys.tool_call_id.has_value());

    auto user = Message::user("User message");
    EXPECT_EQ(user.role, Role::User);
    EXPECT_EQ(user.content, "User message");
    EXPECT_FALSE(user.tool_call_id.has_value());

    auto assistant = Message::assistant("Assistant message");
    EXPECT_EQ(assistant.role, Role::Assistant);
    EXPECT_EQ(assistant.content, "Assistant message");
    EXPECT_FALSE(assistant.tool_call_id.has_value());

    auto tool = Message::tool("Tool result", "call_123");
    EXPECT_EQ(tool.role, Role::Tool);
    EXPECT_EQ(tool.content, "Tool result");
    ASSERT_TRUE(tool.tool_call_id.has_value());
    EXPECT_EQ(*tool.tool_call_id, "call_123");
}

TEST(MessageTest, Equality) {
    auto msg1 = Message::user("Hello");
    auto msg2 = Message::user("Hello");
    auto msg3 = Message::user("World");
    auto msg4 = Message::assistant("Hello");

    EXPECT_EQ(msg1, msg2);
    EXPECT_NE(msg1, msg3);
    EXPECT_NE(msg1, msg4);

    auto tool1 = Message::tool("Result", "id1");
    auto tool2 = Message::tool("Result", "id1");
    auto tool3 = Message::tool("Result", "id2");

    EXPECT_EQ(tool1, tool2);
    EXPECT_NE(tool1, tool3);
}

TEST(RoleTest, RoleToString) {
    EXPECT_EQ(role_to_string(Role::System), "system");
    EXPECT_EQ(role_to_string(Role::User), "user");
    EXPECT_EQ(role_to_string(Role::Assistant), "assistant");
    EXPECT_EQ(role_to_string(Role::Tool), "tool");
}

// ============================================================================
// Error Tests
// ============================================================================

TEST(ErrorTest, Construction) {
    Error err(ErrorCode::InvalidConfig, "Test error");
    EXPECT_EQ(err.code, ErrorCode::InvalidConfig);
    EXPECT_EQ(err.message, "Test error");
    EXPECT_FALSE(err.context.has_value());

    Error err_with_context(ErrorCode::InferenceFailed, "Test error", "Additional context");
    EXPECT_EQ(err_with_context.code, ErrorCode::InferenceFailed);
    EXPECT_EQ(err_with_context.message, "Test error");
    ASSERT_TRUE(err_with_context.context.has_value());
    EXPECT_EQ(*err_with_context.context, "Additional context");
}

TEST(ErrorTest, ToString) {
    Error err(ErrorCode::InvalidConfig, "Configuration is invalid");
    std::string str = err.to_string();
    EXPECT_NE(str.find("100"), std::string::npos);  // Error code
    EXPECT_NE(str.find("Configuration is invalid"), std::string::npos);

    Error err_with_context(ErrorCode::InferenceFailed, "Inference failed", "Out of memory");
    str = err_with_context.to_string();
    EXPECT_NE(str.find("203"), std::string::npos);  // Error code
    EXPECT_NE(str.find("Inference failed"), std::string::npos);
    EXPECT_NE(str.find("Out of memory"), std::string::npos);
}

TEST(ErrorTest, Expected) {
    Expected<int> success = 42;
    EXPECT_TRUE(success.has_value());
    EXPECT_EQ(*success, 42);

    Expected<int> failure = tl::unexpected(Error{ErrorCode::Unknown, "Failed"});
    EXPECT_FALSE(failure.has_value());
    EXPECT_EQ(failure.error().code, ErrorCode::Unknown);
    EXPECT_EQ(failure.error().message, "Failed");
}

// ============================================================================
// SamplingParams Tests
// ============================================================================

TEST(SamplingParamsTest, Defaults) {
    SamplingParams params;
    EXPECT_FLOAT_EQ(params.temperature, 0.7f);
    EXPECT_FLOAT_EQ(params.top_p, 0.9f);
    EXPECT_EQ(params.top_k, 40);
    EXPECT_FLOAT_EQ(params.repeat_penalty, 1.1f);
    EXPECT_EQ(params.repeat_last_n, 64);
    EXPECT_EQ(params.seed, -1);
}

TEST(SamplingParamsTest, Equality) {
    SamplingParams p1;
    SamplingParams p2;
    EXPECT_EQ(p1, p2);

    p2.temperature = 0.5f;
    EXPECT_NE(p1, p2);

    p2 = p1;
    p2.seed = 12345;
    EXPECT_NE(p1, p2);
}

// ============================================================================
// PromptTemplate Tests
// ============================================================================

TEST(PromptTemplateTest, TemplateToString) {
    EXPECT_EQ(template_to_string(PromptTemplate::Llama3), "Llama3");
    EXPECT_EQ(template_to_string(PromptTemplate::ChatML), "ChatML");
    EXPECT_EQ(template_to_string(PromptTemplate::Custom), "Custom");
}

// ============================================================================
// Config Tests
// ============================================================================

TEST(ConfigTest, Defaults) {
    Config config;
    config.model_path = "/path/to/model.gguf";

    EXPECT_EQ(config.context_size, 8192);
    EXPECT_EQ(config.n_gpu_layers, -1);
    EXPECT_TRUE(config.use_mmap);
    EXPECT_FALSE(config.use_mlock);
    EXPECT_EQ(config.prompt_template, PromptTemplate::Llama3);
    EXPECT_EQ(config.max_tokens, 512);
    EXPECT_FALSE(config.system_prompt.has_value());
    EXPECT_FALSE(config.custom_template.has_value());
}

TEST(ConfigTest, ValidationSuccess) {
    Config config;
    config.model_path = "/path/to/model.gguf";

    auto result = config.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(ConfigTest, ValidationEmptyModelPath) {
    Config config;
    config.model_path = "";

    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidModelPath);
}

TEST(ConfigTest, ValidationInvalidContextSize) {
    Config config;
    config.model_path = "/path/to/model.gguf";
    config.context_size = 0;

    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidContextSize);

    config.context_size = -100;
    result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidContextSize);
}

TEST(ConfigTest, ValidationInvalidMaxTokens) {
    Config config;
    config.model_path = "/path/to/model.gguf";
    config.max_tokens = 0;

    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidConfig);
}

TEST(ConfigTest, ValidationCustomTemplateRequired) {
    Config config;
    config.model_path = "/path/to/model.gguf";
    config.prompt_template = PromptTemplate::Custom;

    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidTemplate);

    config.custom_template = "{{role}}: {{content}}";
    result = config.validate();
    EXPECT_TRUE(result.has_value());
}

TEST(ConfigTest, Equality) {
    Config c1;
    c1.model_path = "/path/to/model.gguf";

    Config c2;
    c2.model_path = "/path/to/model.gguf";

    EXPECT_EQ(c1, c2);

    c2.context_size = 4096;
    EXPECT_NE(c1, c2);

    c2 = c1;
    c2.system_prompt = "You are a helpful assistant.";
    EXPECT_NE(c1, c2);
}

// ============================================================================
// TokenUsage Tests
// ============================================================================

TEST(TokenUsageTest, Defaults) {
    TokenUsage usage;
    EXPECT_EQ(usage.prompt_tokens, 0);
    EXPECT_EQ(usage.completion_tokens, 0);
    EXPECT_EQ(usage.total_tokens, 0);
}

TEST(TokenUsageTest, Equality) {
    TokenUsage u1{10, 20, 30};
    TokenUsage u2{10, 20, 30};
    TokenUsage u3{10, 20, 31};

    EXPECT_EQ(u1, u2);
    EXPECT_NE(u1, u3);
}

// ============================================================================
// Metrics Tests
// ============================================================================

TEST(MetricsTest, Defaults) {
    Metrics metrics;
    EXPECT_EQ(metrics.latency_ms.count(), 0);
    EXPECT_EQ(metrics.time_to_first_token_ms.count(), 0);
    EXPECT_DOUBLE_EQ(metrics.tokens_per_second, 0.0);
}

TEST(MetricsTest, Equality) {
    Metrics m1;
    m1.latency_ms = std::chrono::milliseconds(100);
    m1.time_to_first_token_ms = std::chrono::milliseconds(50);
    m1.tokens_per_second = 25.5;

    Metrics m2;
    m2.latency_ms = std::chrono::milliseconds(100);
    m2.time_to_first_token_ms = std::chrono::milliseconds(50);
    m2.tokens_per_second = 25.5;

    EXPECT_EQ(m1, m2);

    m2.tokens_per_second = 30.0;
    EXPECT_NE(m1, m2);
}

// ============================================================================
// Response Tests
// ============================================================================

TEST(ResponseTest, Defaults) {
    Response response;
    EXPECT_TRUE(response.text.empty());
    EXPECT_EQ(response.usage.total_tokens, 0);
    EXPECT_EQ(response.metrics.latency_ms.count(), 0);
    EXPECT_TRUE(response.tool_calls.empty());
}

TEST(ResponseTest, Equality) {
    Response r1;
    r1.text = "Hello";
    r1.usage = TokenUsage{10, 5, 15};
    r1.metrics.latency_ms = std::chrono::milliseconds(100);

    Response r2;
    r2.text = "Hello";
    r2.usage = TokenUsage{10, 5, 15};
    r2.metrics.latency_ms = std::chrono::milliseconds(100);

    EXPECT_EQ(r1, r2);

    r2.text = "World";
    EXPECT_NE(r1, r2);
}

// ============================================================================
// Request Tests
// ============================================================================

TEST(RequestTest, Construction) {
    auto msg = Message::user("Hello");
    Request req(msg);

    EXPECT_EQ(req.message.role, Role::User);
    EXPECT_EQ(req.message.content, "Hello");
    EXPECT_FALSE(req.streaming_callback.has_value());
}

TEST(RequestTest, WithCallback) {
    auto msg = Message::user("Hello");
    bool called = false;
    auto callback = [&called](std::string_view) { called = true; };

    Request req(msg, callback);
    EXPECT_TRUE(req.streaming_callback.has_value());

    if (req.streaming_callback) {
        (*req.streaming_callback)("token");
        EXPECT_TRUE(called);
    }
}
