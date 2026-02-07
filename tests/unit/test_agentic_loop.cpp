#include <gtest/gtest.h>
#include "zoo/engine/agentic_loop.hpp"
#include "mocks/mock_backend.hpp"
#include "fixtures/tool_definitions.hpp"
#include "fixtures/sample_responses.hpp"

using namespace zoo;
using namespace zoo::engine;
using namespace zoo::testing;
using namespace zoo::testing::tools;
using namespace zoo::testing::responses;
using json = nlohmann::json;

class AgenticLoopToolTest : public ::testing::Test {
protected:
    std::shared_ptr<MockBackend> backend;
    std::shared_ptr<HistoryManager> history;
    std::shared_ptr<ToolRegistry> registry;
    Config config;

    void SetUp() override {
        backend = std::make_shared<MockBackend>();
        config.model_path = "/path/to/model.gguf";
        config.context_size = 8192;
        config.max_tokens = 512;
        (void)backend->initialize(config);

        history = std::make_shared<HistoryManager>(config.context_size);
        registry = std::make_shared<ToolRegistry>();

        registry->register_tool("add", "Add two integers", {"a", "b"}, add);
        registry->register_tool("greet", "Greet someone", {"name"}, greet);
        registry->register_tool("get_time", "Get current time", {}, get_time);
    }

    std::unique_ptr<AgenticLoop> make_loop() {
        auto loop = std::make_unique<AgenticLoop>(backend, history, config);
        loop->set_tool_registry(registry);
        return loop;
    }
};

// ============================================================================
// TA-003: Output contains tool call JSON -> detected and parsed
// ============================================================================

TEST_F(AgenticLoopToolTest, ToolCallDetected) {
    auto loop = make_loop();

    // First response: tool call, second response: final answer
    backend->enqueue_response(TOOL_CALL_ADD);
    backend->enqueue_response("The sum is 7.");

    Request req(Message::user("What is 3 + 4?"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "The sum is 7.");
    EXPECT_FALSE(result->tool_calls.empty());
}

// ============================================================================
// TA-004: Output contains no tool call -> response returned directly
// ============================================================================

TEST_F(AgenticLoopToolTest, NoToolCallDirectResponse) {
    auto loop = make_loop();

    backend->enqueue_response(PLAIN_TEXT);

    Request req(Message::user("What is the capital of France?"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, PLAIN_TEXT);
    EXPECT_TRUE(result->tool_calls.empty());
}

// ============================================================================
// TA-005: Tool call detected -> registered handler invoked
// ============================================================================

TEST_F(AgenticLoopToolTest, ToolHandlerInvoked) {
    auto loop = make_loop();

    // Model calls add(3, 4), then gives final answer
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 3, "b": 4}})");
    backend->enqueue_response("The result is 7.");

    Request req(Message::user("Add 3 and 4"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "The result is 7.");
    // Tool call history should contain the tool result
    ASSERT_EQ(result->tool_calls.size(), 1);
    EXPECT_EQ(result->tool_calls[0].role, Role::Tool);
    EXPECT_NE(result->tool_calls[0].content.find("7"), std::string::npos);
}

// ============================================================================
// TA-006: Tool executed -> result injected, inference continues
// ============================================================================

TEST_F(AgenticLoopToolTest, ToolResultInjected) {
    auto loop = make_loop();

    backend->enqueue_response(R"({"name": "greet", "arguments": {"name": "Bob"}})");
    backend->enqueue_response("I greeted Bob for you.");

    Request req(Message::user("Say hello to Bob"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "I greeted Bob for you.");

    // History should contain: user, assistant (tool call), tool result, assistant (final)
    auto messages = history->get_messages();
    EXPECT_GE(messages.size(), 4);

    // Find the tool message
    bool found_tool_msg = false;
    for (const auto& msg : messages) {
        if (msg.role == Role::Tool) {
            found_tool_msg = true;
            EXPECT_NE(msg.content.find("Hello, Bob!"), std::string::npos);
        }
    }
    EXPECT_TRUE(found_tool_msg);
}

// ============================================================================
// TA-007: Multiple sequential tool calls -> all executed in order
// ============================================================================

TEST_F(AgenticLoopToolTest, MultipleSequentialToolCalls) {
    auto loop = make_loop();

    // First tool call, second tool call, then final answer
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 1, "b": 2}})");
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 3, "b": 4}})");
    backend->enqueue_response("Results: 3 and 7.");

    Request req(Message::user("Add 1+2 and 3+4"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "Results: 3 and 7.");
    EXPECT_EQ(result->tool_calls.size(), 2);
}

// ============================================================================
// TA-008: Stop called during tool loop -> inference aborted
// ============================================================================

TEST_F(AgenticLoopToolTest, CancellationDuringToolLoop) {
    auto loop = make_loop();

    // Enqueue a tool call response, but cancel before second iteration
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 1, "b": 2}})");
    backend->enqueue_response("This should not appear.");

    // Cancel immediately after first iteration begins
    loop->cancel();

    Request req(Message::user("Add 1 and 2"));
    auto result = loop->process_request(req);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::RequestCancelled);
}

// ============================================================================
// TA-010: Streaming callback fires during tool-using conversation
// ============================================================================

TEST_F(AgenticLoopToolTest, StreamingWithToolCalls) {
    auto loop = make_loop();

    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 5, "b": 3}})");
    backend->enqueue_response("The answer is 8.");

    std::vector<std::string> streamed_tokens;
    auto callback = [&streamed_tokens](std::string_view token) {
        streamed_tokens.push_back(std::string(token));
    };

    Request req(Message::user("What is 5+3?"), callback);
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    // Streaming callback should have been called during generation
    EXPECT_GT(streamed_tokens.size(), 0);
}

// ============================================================================
// Tool loop limit reached
// ============================================================================

TEST_F(AgenticLoopToolTest, ToolLoopLimitReached) {
    auto loop = make_loop();
    loop->set_max_tool_iterations(2);

    // Both responses are tool calls -> never gets a final answer
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 1, "b": 2}})");
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 3, "b": 4}})");
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 5, "b": 6}})");

    Request req(Message::user("Add all the things"));
    auto result = loop->process_request(req);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolLoopLimitReached);
}

// ============================================================================
// No tool registry -> behaves like before (no tool detection)
// ============================================================================

TEST_F(AgenticLoopToolTest, NoRegistryNoToolDetection) {
    // Create loop without setting tool registry
    auto loop = std::make_unique<AgenticLoop>(backend, history, config);

    // Even if the response looks like a tool call, it's treated as plain text
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 1, "b": 2}})");

    Request req(Message::user("Test"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    // The "tool call" text is returned as-is
    EXPECT_NE(result->text.find("add"), std::string::npos);
    EXPECT_TRUE(result->tool_calls.empty());
}

// ============================================================================
// Zero-arg tool works
// ============================================================================

TEST_F(AgenticLoopToolTest, ZeroArgToolCall) {
    auto loop = make_loop();

    backend->enqueue_response(R"({"name": "get_time", "arguments": {}})");
    backend->enqueue_response("The time is 2024-01-01T00:00:00Z.");

    Request req(Message::user("What time is it?"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "The time is 2024-01-01T00:00:00Z.");
    EXPECT_EQ(result->tool_calls.size(), 1);
}

// ============================================================================
// Tool validation failure with retry
// ============================================================================

TEST_F(AgenticLoopToolTest, ToolValidationFailureThenRetry) {
    auto loop = make_loop();

    // First call has wrong args, second has correct args, third is final answer
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": "three", "b": 4}})");
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 3, "b": 4}})");
    backend->enqueue_response("The answer is 7.");

    Request req(Message::user("Add 3 + 4"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->text, "The answer is 7.");
    // Should have both error and success tool call entries
    EXPECT_GE(result->tool_calls.size(), 2);
}

// ============================================================================
// Tool retries exhausted
// ============================================================================

TEST_F(AgenticLoopToolTest, ToolRetriesExhausted) {
    auto loop = make_loop();

    // Three consecutive wrong-type tool calls (max retries = 2, so 3rd fails)
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": "x", "b": 4}})");
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": "y", "b": 4}})");
    backend->enqueue_response(R"({"name": "add", "arguments": {"a": "z", "b": 4}})");

    Request req(Message::user("Add stuff"));
    auto result = loop->process_request(req);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolRetriesExhausted);
}

// ============================================================================
// Metrics across tool loop
// ============================================================================

TEST_F(AgenticLoopToolTest, MetricsAccumulateAcrossLoop) {
    auto loop = make_loop();

    backend->enqueue_response(R"({"name": "add", "arguments": {"a": 1, "b": 2}})");
    backend->enqueue_response("Done.");

    Request req(Message::user("Test"));
    auto result = loop->process_request(req);

    ASSERT_TRUE(result.has_value());
    // Prompt tokens should accumulate from both iterations
    EXPECT_GT(result->usage.prompt_tokens, 0);
    EXPECT_GT(result->usage.total_tokens, 0);
}
