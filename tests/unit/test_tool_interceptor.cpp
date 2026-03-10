/**
 * @file test_tool_interceptor.cpp
 * @brief Unit tests for streamed tool-call interception behavior.
 */

#include "zoo/internal/tools/interceptor.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using zoo::TokenAction;
using zoo::tools::ToolCallInterceptor;

/// Simulates token streaming through a `ToolCallInterceptor`.
static ToolCallInterceptor::Result simulate_tokens(const std::vector<std::string>& tokens,
                                                   std::string* streamed_output = nullptr) {
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
        if (action == TokenAction::Stop)
            break;
    }

    return interceptor.finalize();
}

TEST(ToolCallInterceptorTest, DetectsToolCallAndStopsGeneration) {
    std::vector<std::string> tokens = {
        "I'll add those for you.\n",
        "{\"name\":",
        " \"add\",",
        " \"arguments\":",
        " {\"a\": 3,",
        " \"b\": 4}",
        "}",
        " Here is the result..." // should never be reached
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
    std::vector<std::string> tokens = {"The capital of France", " is Paris."};

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(streamed, "The capital of France is Paris.");
    EXPECT_EQ(result.visible_text, "The capital of France is Paris.");
}

TEST(ToolCallInterceptorTest, ToolCallOnlyNoPrefix) {
    std::vector<std::string> tokens = {"{\"name\": \"greet\",",
                                       " \"arguments\": {\"name\": \"Alice\"}}"};

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
    std::vector<std::string> tokens = {"Here is data: ", "{\"key\": \"value\",", " \"count\": 42}",
                                       " and more text."};

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(streamed, "Here is data: {\"key\": \"value\", \"count\": 42} and more text.");
}

TEST(ToolCallInterceptorTest, NonToolJsonFollowedByToolCall) {
    std::vector<std::string> tokens = {"Data: {\"x\": 1}", " Now calling: ",
                                       "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}"};

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
        "Hello! {\"name\": \"greet\", \"arguments\": {\"name\": \"Bob\"}}"};

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
    std::vector<std::string> tokens = {"Here: ", "{\"name\": \"add\", \"arg"};

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    // Incomplete JSON — should be flushed as visible text
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.visible_text, "Here: {\"name\": \"add\", \"arg");
}

TEST(ToolCallInterceptorTest, CompleteToolCallAtEOS) {
    // Model generates a complete tool call and then EOS (no Stop needed)
    std::vector<std::string> tokens = {"{\"name\": \"test\", \"arguments\": {}}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "test");
}

// ============================================================================
// Full text tracking
// ============================================================================

TEST(ToolCallInterceptorTest, FullTextIncludesEverything) {
    std::vector<std::string> tokens = {"Prefix ",
                                       "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.full_text, "Prefix {\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}");
}

TEST(ToolCallInterceptorTest, NoCallbackStillWorks) {
    std::vector<std::string> tokens = {"{\"name\": \"test\", \"arguments\": {}}"};

    // No user callback — should still detect tool call
    ToolCallInterceptor interceptor;
    auto callback = interceptor.make_callback();
    for (const auto& t : tokens)
        callback(t);
    auto result = interceptor.finalize();

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "test");
}

// ============================================================================
// Preserves tool call ID
// ============================================================================

TEST(ToolCallInterceptorTest, PreservesToolCallId) {
    std::vector<std::string> tokens = {
        "{\"name\": \"search\", \"id\": \"call_42\", \"arguments\": {\"q\": \"test\"}}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->id, "call_42");
    EXPECT_EQ(result.tool_call->name, "search");
}

// ============================================================================
// Multiple non-tool JSON objects before tool call
// ============================================================================

TEST(ToolCallInterceptorTest, MultipleJsonBeforeToolCall) {
    std::vector<std::string> tokens = {"{\"info\": 1} ", "{\"info\": 2} ",
                                       "{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}"};

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

// ============================================================================
// Full text tracking without a tool call
// ============================================================================

TEST(ToolCallInterceptorTest, FullTextMatchesVisibleTextWhenNoToolCall) {
    // When there is no tool call, full_text should equal visible_text exactly
    std::vector<std::string> tokens = {"The answer is ", "42."};

    auto result = simulate_tokens(tokens);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.full_text, "The answer is 42.");
    EXPECT_EQ(result.full_text, result.visible_text);
}

// ============================================================================
// Argument type coverage
// ============================================================================

TEST(ToolCallInterceptorTest, ToolCallWithBooleanArguments) {
    std::vector<std::string> tokens = {"{\"name\": \"set_flag\",",
                                       " \"arguments\": {\"enabled\": true, \"verbose\": false}}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "set_flag");
    EXPECT_EQ(result.tool_call->arguments["enabled"], true);
    EXPECT_EQ(result.tool_call->arguments["verbose"], false);
}

TEST(ToolCallInterceptorTest, ToolCallWithNullArgument) {
    std::vector<std::string> tokens = {"{\"name\": \"process\",",
                                       " \"arguments\": {\"value\": null, \"label\": \"test\"}}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "process");
    EXPECT_TRUE(result.tool_call->arguments["value"].is_null());
    EXPECT_EQ(result.tool_call->arguments["label"], "test");
}

TEST(ToolCallInterceptorTest, ToolCallWithNestedObjectArguments) {
    std::vector<std::string> tokens = {"{\"name\": \"create\",",
                                       " \"arguments\": {\"config\": {\"x\": 1, \"y\": 2}}}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "create");
    EXPECT_EQ(result.tool_call->arguments["config"]["x"], 1);
    EXPECT_EQ(result.tool_call->arguments["config"]["y"], 2);
}

// ============================================================================
// Non-tool JSON immediately followed by tool call in one token
// ============================================================================

TEST(ToolCallInterceptorTest, NonToolJsonAndToolCallInSingleToken) {
    // Both the non-tool JSON and the tool call JSON appear in one token.
    // This exercises the process_normal(token.substr(i+1)) recursion after
    // flushing the non-tool JSON.
    std::vector<std::string> tokens = {
        "{\"x\": 1}{\"name\": \"add\", \"arguments\": {\"a\": 5, \"b\": 6}}"};

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 5);
    EXPECT_EQ(result.tool_call->arguments["b"], 6);
    // The non-tool JSON should be in visible text; the tool call JSON should not
    EXPECT_EQ(streamed, "{\"x\": 1}");
    EXPECT_EQ(result.visible_text, "{\"x\": 1}");
}

// ============================================================================
// Whitespace-only visible prefix
// ============================================================================

TEST(ToolCallInterceptorTest, WhitespaceOnlyPrefixBeforeToolCall) {
    // Some models emit a newline before tool call JSON — verify the newline
    // is treated as visible text and the tool call is still detected.
    std::vector<std::string> tokens = {"\n", "{\"name\": \"ping\", \"arguments\": {}}"};

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "ping");
    EXPECT_EQ(streamed, "\n");
    EXPECT_EQ(result.visible_text, "\n");
}

// ============================================================================
// Closing brace arrives as its own token
// ============================================================================

TEST(ToolCallInterceptorTest, ClosingBraceInSeparateToken) {
    // The final '}' of the tool call arrives as a separate single-character
    // token, distinct from the rest of the JSON body.
    std::vector<std::string> tokens = {"{\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}",
                                       "}"};

    auto result = simulate_tokens(tokens);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 1);
    EXPECT_EQ(result.tool_call->arguments["b"], 2);
    EXPECT_TRUE(result.visible_text.empty());
}

// ============================================================================
// Empty JSON object is treated as non-tool and flushed as visible text
// ============================================================================

TEST(ToolCallInterceptorTest, EmptyJsonObjectPassesThroughAsText) {
    // {} has no "name" or "arguments" — should be flushed as plain text
    std::vector<std::string> tokens = {"Result: ", "{}", " done."};

    std::string streamed;
    auto result = simulate_tokens(tokens, &streamed);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(streamed, "Result: {} done.");
    EXPECT_EQ(result.visible_text, "Result: {} done.");
}

// ============================================================================
// Tokens after Stop are not processed
// ============================================================================

TEST(ToolCallInterceptorTest, TokensAfterStopAreIgnored) {
    // After a tool call is detected and Stop is returned, any subsequent tokens
    // that the caller might feed in should not affect the result.
    std::vector<std::string> all_tokens = {
        "{\"name\": \"first\", \"arguments\": {}}",
        "{\"name\": \"second\", \"arguments\": {}}" // should never be processed
    };

    // simulate_tokens already breaks on Stop, but let's also verify via
    // explicit manual invocation that only the first tool call is captured
    ToolCallInterceptor interceptor;
    auto callback = interceptor.make_callback();

    TokenAction action = callback(all_tokens[0]);
    EXPECT_EQ(action, TokenAction::Stop);

    // Feeding additional tokens after Stop — simulate_tokens wouldn't do this,
    // but if a caller ignores the Stop signal and keeps feeding tokens, the
    // interceptor should still only report the first tool call
    callback(all_tokens[1]);

    auto result = interceptor.finalize();
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "first");
}

// ============================================================================
// Full text with non-tool JSON accumulates all tokens
// ============================================================================

TEST(ToolCallInterceptorTest, FullTextIncludesNonToolJson) {
    // full_text should be the raw concatenation of all tokens regardless of
    // whether the content is a tool call or plain JSON
    std::vector<std::string> tokens = {"Prefix ", "{\"info\": 99}", " suffix"};

    auto result = simulate_tokens(tokens);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.full_text, "Prefix {\"info\": 99} suffix");
    EXPECT_EQ(result.full_text, result.visible_text);
}
