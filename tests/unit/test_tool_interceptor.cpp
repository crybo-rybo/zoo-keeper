#include <gtest/gtest.h>
#include "zoo/tools/interceptor.hpp"
#include <string>
#include <vector>

using zoo::TokenAction;
using zoo::tools::ToolCallInterceptor;

// Helper: simulate token-by-token generation through the interceptor
static ToolCallInterceptor::Result simulate_tokens(
    const std::vector<std::string>& tokens,
    std::string* streamed_output = nullptr
) {
    std::optional<std::function<void(std::string_view)>> user_cb;
    if (streamed_output) {
        user_cb = [streamed_output](std::string_view sv) {
            streamed_output->append(sv.data(), sv.size());
        };
    }

    ToolCallInterceptor interceptor(std::move(user_cb));
    auto callback = interceptor.make_callback();

    for (const auto& token : tokens) {
        TokenAction action = callback(token);
        if (action == TokenAction::Stop) break;
    }

    return interceptor.finalize();
}

// ============================================================================
// Basic detection
// ============================================================================

TEST(ToolCallInterceptorTest, DetectsToolCallAndStopsGeneration) {
    std::vector<std::string> tokens = {
        "I'll add those for you.\n",
        "{\"name\":",
        " \"add\",",
        " \"arguments\":",
        " {\"a\": 3,",
        " \"b\": 4}",
        "}",
        " Here is the result..."  // should never be reached
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 3);
    EXPECT_EQ(result.tool_call->arguments["b"], 4);

    // User should only see the text before the tool call
    EXPECT_EQ(streamed, "I'll add those for you.\n");
    EXPECT_EQ(result.visible_text, "I'll add those for you.\n");
}

TEST(ToolCallInterceptorTest, NoToolCallPassesThrough) {
    std::vector<std::string> tokens = {
        "The capital of France",
        " is Paris."
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(streamed, "The capital of France is Paris.");
    EXPECT_EQ(result.visible_text, "The capital of France is Paris.");
}

TEST(ToolCallInterceptorTest, ToolCallOnlyNoPrefix) {
    std::vector<std::string> tokens = {
        "{\"name\": \"greet\",",
        " \"arguments\": {\"name\": \"Alice\"}}"
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "greet");
    EXPECT_EQ(result.tool_call->arguments["name"], "Alice");
    EXPECT_TRUE(streamed.empty());
    EXPECT_TRUE(result.visible_text.empty());
}

// ============================================================================
// Non-tool JSON handling
// ============================================================================

TEST(ToolCallInterceptorTest, NonToolJsonPassesThrough) {
    std::vector<std::string> tokens = {
        "Here is data: ",
        "{\"key\": \"value\",",
        " \"count\": 42}",
        " and more text."
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(streamed, "Here is data: {\"key\": \"value\", \"count\": 42} and more text.");
}

TEST(ToolCallInterceptorTest, NonToolJsonFollowedByToolCall) {
    std::vector<std::string> tokens = {
        "Data: {\"x\": 1}",
        " Now calling: ",
        "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}"
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    // The non-tool JSON and text before should be visible
    EXPECT_EQ(streamed, "Data: {\"x\": 1} Now calling: ");
}

// ============================================================================
// Edge cases: braces in strings
// ============================================================================

TEST(ToolCallInterceptorTest, BracesInsideStrings) {
    std::vector<std::string> tokens = {
        "{\"name\": \"test\",",
        " \"arguments\": {\"pattern\": \"{hello}\"}}",
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "test");
    EXPECT_EQ(result.tool_call->arguments["pattern"], "{hello}");
}

TEST(ToolCallInterceptorTest, EscapedQuotesInStrings) {
    std::vector<std::string> tokens = {
        "{\"name\": \"test\",",
        " \"arguments\": {\"msg\": \"say \\\"hi\\\"\"}}",
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "test");
}

// ============================================================================
// Token granularity
// ============================================================================

TEST(ToolCallInterceptorTest, SingleCharacterTokens) {
    // Simulate very fine-grained tokenization
    std::string tool_call = R"({"name": "add", "arguments": {"a": 1, "b": 2}})";
    std::vector<std::string> tokens;
    for (char c : tool_call) {
        tokens.emplace_back(1, c);
    }

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 1);
    EXPECT_EQ(result.tool_call->arguments["b"], 2);
}

TEST(ToolCallInterceptorTest, EntireOutputInOneToken) {
    std::vector<std::string> tokens = {
        "Hello! {\"name\": \"greet\", \"arguments\": {\"name\": \"Bob\"}}"
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "greet");
    EXPECT_EQ(streamed, "Hello! ");
}

// ============================================================================
// Generation ends during buffering (EOS mid-tool-call)
// ============================================================================

TEST(ToolCallInterceptorTest, IncompleteToolCallAtEOS) {
    // Model stops generating mid-JSON (hit EOS or max tokens)
    std::vector<std::string> tokens = {
        "Here: ",
        "{\"name\": \"add\", \"arg"
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    // Incomplete JSON — should be flushed as visible text
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.visible_text, "Here: {\"name\": \"add\", \"arg");
}

TEST(ToolCallInterceptorTest, CompleteToolCallAtEOS) {
    // Model generates a complete tool call and then EOS (no Stop needed)
    std::vector<std::string> tokens = {
        "{\"name\": \"test\", \"arguments\": {}}"
    };

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "test");
}

// ============================================================================
// Full text tracking
// ============================================================================

TEST(ToolCallInterceptorTest, FullTextIncludesEverything) {
    std::vector<std::string> tokens = {
        "Prefix ",
        "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}"
    };

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.full_text,
        "Prefix {\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}");
}

TEST(ToolCallInterceptorTest, NoCallbackStillWorks) {
    std::vector<std::string> tokens = {
        "{\"name\": \"test\", \"arguments\": {}}"
    };

    // No user callback — should still detect tool call
    ToolCallInterceptor interceptor;
    auto callback = interceptor.make_callback();
    for (const auto& t : tokens) callback(t);
    auto result = interceptor.finalize();

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "test");
}

// ============================================================================
// Preserves tool call ID
// ============================================================================

TEST(ToolCallInterceptorTest, PreservesToolCallId) {
    std::vector<std::string> tokens = {
        "{\"name\": \"search\", \"id\": \"call_42\", \"arguments\": {\"q\": \"test\"}}"
    };

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->id, "call_42");
    EXPECT_EQ(result.tool_call->name, "search");
}

// ============================================================================
// Multiple non-tool JSON objects before tool call
// ============================================================================

TEST(ToolCallInterceptorTest, MultipleJsonBeforeToolCall) {
    std::vector<std::string> tokens = {
        "{\"info\": 1} ",
        "{\"info\": 2} ",
        "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}"
    };

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(streamed, "{\"info\": 1} {\"info\": 2} ");
}

TEST(ToolCallInterceptorTest, EmptyInput) {
    auto result = simulate_tokens({});
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_TRUE(result.visible_text.empty());
    EXPECT_TRUE(result.full_text.empty());
}
