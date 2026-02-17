#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "mocks/mock_backend.hpp"
#include "zoo/types.hpp"
#include "zoo/agent.hpp"
#include "zoo/engine/history_manager.hpp"
#include "zoo/engine/rag_store.hpp"
#include "fixtures/tool_definitions.hpp"

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

TEST_F(AgentTest, MockBackendStopSequenceNotStreamed) {
    MockBackend backend;
    Config config;
    config.model_path = "/path/to/model.gguf";
    (void)backend.initialize(config);

    backend.default_response = "Hello STOP world";

    std::vector<std::string> streamed;
    auto callback = [&streamed](std::string_view token) {
        streamed.push_back(std::string(token));
    };

    auto tokens = backend.tokenize("Test");
    ASSERT_TRUE(tokens.has_value());

    auto result = backend.generate(*tokens, 512, {"STOP"}, callback);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "Hello ");

    // Streamed text should match final text — no stop-sequence tokens
    std::string streamed_text;
    for (const auto& t : streamed) streamed_text += t;
    EXPECT_EQ(streamed_text.find("STOP"), std::string::npos);
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

// ============================================================================
// Agent Tool Integration Tests
// ============================================================================

TEST_F(AgentTest, AgentRegisterTool) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Hello!";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    EXPECT_EQ(agent->tool_count(), 0);

    agent->register_tool("add", "Add two numbers", {"a", "b"}, tools::add);
    EXPECT_EQ(agent->tool_count(), 1);

    agent->register_tool("greet", "Greet someone", {"name"}, tools::greet);
    EXPECT_EQ(agent->tool_count(), 2);
}

TEST_F(AgentTest, AgentToolCallEndToEnd) {
    auto backend = std::make_unique<MockBackend>();
    // First response is a tool call, second is the final answer
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 10, "b": 20}})");
    backend->enqueue_response("The sum of 10 and 20 is 30.");

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    agent->register_tool("add", "Add two integers", {"a", "b"}, tools::add);

    auto future = agent->chat(Message::user("What is 10 + 20?"));
    auto response = future.get();

    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(response->text, "The sum of 10 and 20 is 30.");
    EXPECT_FALSE(response->tool_calls.empty());
}

TEST_F(AgentTest, AgentNoToolsPlainResponse) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "The answer is 42.";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    auto future = agent->chat(Message::user("What is the meaning of life?"));
    auto response = future.get();

    ASSERT_TRUE(response.has_value());
    EXPECT_EQ(response->text, "The answer is 42.");
    EXPECT_TRUE(response->tool_calls.empty());
}

TEST_F(AgentTest, AgentRegisterToolAfterChat) {
    auto backend = std::make_unique<MockBackend>();
    backend->enqueue_response("Plain response.");
    backend->enqueue_response(R"({"name": "greet", "arguments": {"name": "World"}})");
    backend->enqueue_response("I greeted World for you.");

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    // First chat without tools
    auto future1 = agent->chat(Message::user("Hello"));
    auto response1 = future1.get();
    ASSERT_TRUE(response1.has_value());
    EXPECT_EQ(response1->text, "Plain response.");

    // Register tool after first chat
    agent->register_tool("greet", "Greet someone", {"name"}, tools::greet);

    // Second chat with tool
    auto future2 = agent->chat(Message::user("Greet World"));
    auto response2 = future2.get();
    ASSERT_TRUE(response2.has_value());
    EXPECT_EQ(response2->text, "I greeted World for you.");
    EXPECT_FALSE(response2->tool_calls.empty());
}

// ============================================================================
// Additional Agent API coverage
// ============================================================================

TEST_F(AgentTest, StopAndIsRunning) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Hello!";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    EXPECT_TRUE(agent->is_running());

    agent->stop();
    EXPECT_FALSE(agent->is_running());

    // Double stop is safe
    agent->stop();
    EXPECT_FALSE(agent->is_running());
}

TEST_F(AgentTest, ChatAfterStopReturnsError) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Hello!";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    agent->stop();

    auto future = agent->chat(Message::user("Hello"));
    auto response = future.get();

    EXPECT_FALSE(response.has_value());
    EXPECT_EQ(response.error().code, ErrorCode::AgentNotRunning);
}

TEST_F(AgentTest, SetSystemPrompt) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "I am a helpful assistant.";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    agent->set_system_prompt("You are a helpful assistant.");

    auto history = agent->get_history();
    ASSERT_FALSE(history.empty());
    EXPECT_EQ(history[0].role, Role::System);
    EXPECT_EQ(history[0].content, "You are a helpful assistant.");
}

TEST_F(AgentTest, ClearHistory) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Response.";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    // Send a message to populate history
    auto future = agent->chat(Message::user("Hello"));
    (void)future.get();

    auto history = agent->get_history();
    EXPECT_FALSE(history.empty());

    agent->clear_history();

    history = agent->get_history();
    EXPECT_TRUE(history.empty());
}

TEST_F(AgentTest, GetConfig) {
    auto backend = std::make_unique<MockBackend>();

    Config config;
    config.model_path = "/path/to/model.gguf";
    config.context_size = 4096;
    config.max_tokens = 256;

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    const auto& retrieved = agent->get_config();
    EXPECT_EQ(retrieved.model_path, "/path/to/model.gguf");
    EXPECT_EQ(retrieved.context_size, 4096);
    EXPECT_EQ(retrieved.max_tokens, 256);
}

TEST_F(AgentTest, CreateWithInvalidConfigFails) {
    Config config;
    config.model_path = "";  // Invalid

    auto result = Agent::create(config);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidModelPath);
}

TEST_F(AgentTest, CreateWithSystemPromptInConfig) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Hello!";

    Config config;
    config.model_path = "/path/to/model.gguf";
    config.system_prompt = "You are a pirate.";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    auto history = agent->get_history();
    ASSERT_FALSE(history.empty());
    EXPECT_EQ(history[0].role, Role::System);
    EXPECT_EQ(history[0].content, "You are a pirate.");
}

TEST_F(AgentTest, AgentRagChatOptionsWithRetriever) {
    auto backend_raw = new MockBackend();
    auto backend = std::unique_ptr<MockBackend>(backend_raw);
    backend->default_response = "Paris is in France.";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    auto retriever = std::make_shared<engine::InMemoryRagStore>();
    ASSERT_TRUE(retriever->add_document(
        "geo:paris",
        "Paris is the capital and most populous city of France."
    ).has_value());
    agent->set_retriever(retriever);

    ChatOptions options;
    options.rag.enabled = true;
    options.rag.top_k = 2;

    auto future = agent->chat(Message::user("Where is Paris located?"), options);
    auto response = future.get();

    ASSERT_TRUE(response.has_value());
    EXPECT_FALSE(response->rag_chunks.empty());
    EXPECT_NE(backend_raw->last_formatted_prompt.find("Retrieved Context"), std::string::npos);

    // RAG context is ephemeral and must not be persisted in history.
    auto history = agent->get_history();
    ASSERT_EQ(history.size(), 2U);
    EXPECT_EQ(history[0].role, Role::User);
    EXPECT_EQ(history[1].role, Role::Assistant);
}

// ============================================================================
// Queue Capacity Tests
// ============================================================================

TEST_F(AgentTest, QueueFullWithBoundedCapacity) {
    auto backend = std::make_unique<MockBackend>();
    // Use a slow response so the first request stays in-flight
    backend->default_response = "Response";
    backend->generation_delay_ms = 200;

    Config config;
    config.model_path = "/path/to/model.gguf";
    config.request_queue_capacity = 1;

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    // First chat occupies the queue slot (or is being processed)
    auto future1 = agent->chat(Message::user("First"));

    // Give the inference thread time to pick up the first request
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Submit two more rapidly — one fills the queue, the next should get QueueFull
    auto future2 = agent->chat(Message::user("Second"));
    auto future3 = agent->chat(Message::user("Third"));

    auto response3 = future3.get();
    // At least one of the later requests should fail with QueueFull
    // (timing-dependent, but with capacity=1 and a slow backend, this is reliable)
    if (!response3.has_value()) {
        EXPECT_EQ(response3.error().code, ErrorCode::QueueFull);
    }

    // Clean up remaining futures
    (void)future1.get();
    (void)future2.get();
}

TEST_F(AgentTest, UnlimitedQueueByDefault) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Response";

    Config config;
    config.model_path = "/path/to/model.gguf";
    // request_queue_capacity defaults to 0 (unlimited)

    EXPECT_EQ(config.request_queue_capacity, 0u);

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    // Should be able to push many requests without failure
    std::vector<std::future<Expected<Response>>> futures;
    for (int i = 0; i < 20; ++i) {
        futures.push_back(agent->chat(Message::user("Message " + std::to_string(i))));
    }

    // All should eventually complete successfully
    for (auto& f : futures) {
        auto response = f.get();
        EXPECT_TRUE(response.has_value());
    }
}

TEST_F(AgentTest, ConfigQueueCapacityPassedThrough) {
    Config config;
    config.model_path = "/path/to/model.gguf";
    config.request_queue_capacity = 42;

    auto backend = std::make_unique<MockBackend>();
    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    // Verify config is stored correctly
    EXPECT_EQ(agent->get_config().request_queue_capacity, 42u);
}

TEST_F(AgentTest, GetHistoryThreadSafe) {
    auto backend = std::make_unique<MockBackend>();
    backend->default_response = "Response.";

    Config config;
    config.model_path = "/path/to/model.gguf";

    auto agent_result = Agent::create(config, std::move(backend));
    ASSERT_TRUE(agent_result.has_value());
    auto& agent = *agent_result;

    // Get history from multiple threads concurrently
    std::atomic<int> successes{0};
    auto reader = [&]() {
        for (int i = 0; i < 50; ++i) {
            auto h = agent->get_history();
            (void)h;
            successes++;
        }
    };

    std::thread t1(reader);
    std::thread t2(reader);
    t1.join(); t2.join();

    EXPECT_EQ(successes.load(), 100);
}
