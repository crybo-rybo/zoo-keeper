#include <gtest/gtest.h>
#include "mocks/mock_backend.hpp"
#include "zoo/types.hpp"
#include "zoo/engine/history_manager.hpp"

using namespace zoo;
using namespace zoo::testing;

class AgentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for Agent tests
    }

    void TearDown() override {
        // Cleanup
    }
};

// ============================================================================
// MockBackend Unit Tests
// ============================================================================

TEST_F(AgentTest, MockBackendInitialization) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";

    auto result = backend.initialize(config);
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(backend.initialized);
}

TEST_F(AgentTest, MockBackendTokenization) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    auto tokens = backend.tokenize("Hello, world!");
    ASSERT_TRUE(tokens.has_value());
    EXPECT_GT(tokens->size(), 0);
}

TEST_F(AgentTest, MockBackendGeneration) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.default_response = "Test response";

    auto prompt_tokens = backend.tokenize("Hello");
    ASSERT_TRUE(prompt_tokens.has_value());

    auto result = backend.generate(*prompt_tokens, 512, {});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Test response");
}

TEST_F(AgentTest, MockBackendStreamingCallback) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.default_response = "Hello world test";
    backend.mode = MockBackend::ResponseMode::TokenByToken;

    std::vector<std::string> streamed;
    auto callback = [&streamed](std::string_view token) {
        streamed.push_back(std::string(token));
    };

    auto prompt_tokens = backend.tokenize("Test");
    ASSERT_TRUE(prompt_tokens.has_value());

    auto result = backend.generate(*prompt_tokens, 512, {}, callback);
    ASSERT_TRUE(result.has_value());

    EXPECT_GT(streamed.size(), 0);
    EXPECT_EQ(backend.token_callback_count, streamed.size());
}

TEST_F(AgentTest, MockBackendErrorInjection) {
    MockBackend backend;
    backend.should_fail_initialize = true;
    backend.error_message = "Simulated initialization failure";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto result = backend.initialize(config);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::BackendInitFailed);
    EXPECT_EQ(result.error().message, "Simulated initialization failure");
}

TEST_F(AgentTest, MockBackendGenerateError) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.should_fail_generate = true;
    backend.error_message = "Simulated generation failure";

    auto prompt_tokens = backend.tokenize("Test");
    ASSERT_TRUE(prompt_tokens.has_value());

    auto result = backend.generate(*prompt_tokens, 512, {});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InferenceFailed);
}

TEST_F(AgentTest, MockBackendKVCacheTracking) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    EXPECT_EQ(backend.get_kv_cache_token_count(), 0);

    auto prompt_tokens = backend.tokenize("Test prompt");
    ASSERT_TRUE(prompt_tokens.has_value());

    (void)backend.generate(*prompt_tokens, 512, {});
    EXPECT_GT(backend.get_kv_cache_token_count(), 0);

    backend.clear_kv_cache();
    EXPECT_EQ(backend.get_kv_cache_token_count(), 0);
}

TEST_F(AgentTest, MockBackendResponseQueue) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.enqueue_response("First response");
    backend.enqueue_response("Second response");

    auto tokens = backend.tokenize("Test");
    ASSERT_TRUE(tokens.has_value());

    auto result1 = backend.generate(*tokens, 512, {});
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(*result1, "First response");

    auto result2 = backend.generate(*tokens, 512, {});
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(*result2, "Second response");

    // Falls back to default
    auto result3 = backend.generate(*tokens, 512, {});
    ASSERT_TRUE(result3.has_value());
    EXPECT_EQ(*result3, backend.default_response);
}

TEST_F(AgentTest, MockBackendStopSequences) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.default_response = "Hello STOP world";

    auto tokens = backend.tokenize("Test");
    ASSERT_TRUE(tokens.has_value());

    auto result = backend.generate(*tokens, 512, {"STOP"});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hello ");
}

TEST_F(AgentTest, MockBackendReset) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.enqueue_response("Test");
    auto tokens = backend.tokenize("Test");
    ASSERT_TRUE(tokens.has_value());
    (void)backend.generate(*tokens, 512, {});

    EXPECT_TRUE(backend.initialized);
    EXPECT_GT(backend.kv_cache_tokens, 0);

    backend.reset();

    EXPECT_FALSE(backend.initialized);
    EXPECT_EQ(backend.kv_cache_tokens, 0);
    EXPECT_EQ(backend.token_callback_count, 0);
}

// ============================================================================
// Integration Tests (Components Working Together)
// ============================================================================

TEST_F(AgentTest, HistoryManagerWithFormatPrompt) {
    // Test that HistoryManager and MockBackend::format_prompt work together
    engine::HistoryManager history(8192);
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    history.set_system_prompt("You are a helpful assistant.");
    (void)history.add_message(Message::user("Hello!"));

    auto formatted = backend.format_prompt(history.get_messages());
    ASSERT_TRUE(formatted.has_value());
    EXPECT_NE(formatted->find("You are a helpful assistant."), std::string::npos);
    EXPECT_NE(formatted->find("Hello!"), std::string::npos);
}

TEST_F(AgentTest, FullPipelineSimulation) {
    // Simulate the full agent pipeline without actual Agent class
    engine::HistoryManager history(8192);
    MockBackend backend;

    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);
    backend.default_response = "Hello! How can I assist you today?";

    // Setup
    history.set_system_prompt("You are a helpful AI assistant.");

    // User turn
    (void)history.add_message(Message::user("Hi there!"));
    auto formatted = backend.format_prompt(history.get_messages());
    ASSERT_TRUE(formatted.has_value());

    auto tokens = backend.tokenize(*formatted);
    ASSERT_TRUE(tokens.has_value());

    auto response = backend.generate(*tokens, 512, {});
    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(*response, "Hello! How can I assist you today?");

    // Add assistant response and finalize
    (void)history.add_message(Message::assistant(*response));
    backend.finalize_response(history.get_messages());

    EXPECT_EQ(history.get_messages().size(), 3);  // System + User + Assistant
}

TEST_F(AgentTest, MultiTurnConversationSimulation) {
    // Simulate multiple conversation turns
    engine::HistoryManager history(8192);
    MockBackend backend;

    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.enqueue_response("Paris is the capital of France.");
    backend.enqueue_response("Approximately 2.2 million people.");

    // Turn 1
    (void)history.add_message(Message::user("What is the capital of France?"));
    auto formatted1 = backend.format_prompt(history.get_messages());
    ASSERT_TRUE(formatted1.has_value());
    auto tokens1 = backend.tokenize(*formatted1);
    ASSERT_TRUE(tokens1.has_value());
    auto response1 = backend.generate(*tokens1, 512, {});
    ASSERT_TRUE(response1.has_value());
    (void)history.add_message(Message::assistant(*response1));
    backend.finalize_response(history.get_messages());

    // Turn 2
    (void)history.add_message(Message::user("What is its population?"));
    auto formatted2 = backend.format_prompt(history.get_messages());
    ASSERT_TRUE(formatted2.has_value());
    auto tokens2 = backend.tokenize(*formatted2);
    ASSERT_TRUE(tokens2.has_value());
    auto response2 = backend.generate(*tokens2, 512, {});
    ASSERT_TRUE(response2.has_value());
    (void)history.add_message(Message::assistant(*response2));
    backend.finalize_response(history.get_messages());

    EXPECT_EQ(history.get_messages().size(), 4);
    EXPECT_FALSE(history.is_context_exceeded());
}
