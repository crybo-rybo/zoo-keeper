#include <gtest/gtest.h>
#include "zoo/engine/error_recovery.hpp"
#include "zoo/engine/tool_call_parser.hpp"
#include "fixtures/tool_definitions.hpp"
#include "fixtures/sample_responses.hpp"

using namespace zoo;
using namespace zoo::engine;
using namespace zoo::testing::tools;
using namespace zoo::testing::responses;
using json = nlohmann::json;

class ErrorRecoveryTest : public ::testing::Test {
protected:
    ToolRegistry registry;
    ErrorRecovery recovery{2};

    void SetUp() override {
        registry.register_tool("add", "Add two integers", {"a", "b"}, add);
        registry.register_tool("greet", "Greet someone", {"name"}, greet);
        registry.register_tool("multiply", "Multiply doubles", {"a", "b"}, multiply);
    }
};

// ============================================================================
// ER-001: Valid arguments -> validation passes
// ============================================================================

TEST_F(ErrorRecoveryTest, ValidArgsPass) {
    ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}, {"b", 4}};

    auto error = recovery.validate_args(tc, registry);
    EXPECT_TRUE(error.empty());
}

TEST_F(ErrorRecoveryTest, ValidStringArgs) {
    ToolCall tc;
    tc.name = "greet";
    tc.arguments = {{"name", "Alice"}};

    auto error = recovery.validate_args(tc, registry);
    EXPECT_TRUE(error.empty());
}

// ============================================================================
// ER-002: Missing required argument -> validation fails
// ============================================================================

TEST_F(ErrorRecoveryTest, MissingRequiredArg) {
    ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}};  // missing "b"

    auto error = recovery.validate_args(tc, registry);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("Missing required argument"), std::string::npos);
    EXPECT_NE(error.find("b"), std::string::npos);
}

// ============================================================================
// ER-003: Wrong argument type -> validation fails
// ============================================================================

TEST_F(ErrorRecoveryTest, WrongArgType) {
    ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", "not_a_number"}, {"b", 4}};

    auto error = recovery.validate_args(tc, registry);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("wrong type"), std::string::npos);
}

// ============================================================================
// ER-005: Self-correction succeeds on retry
// ============================================================================

TEST_F(ErrorRecoveryTest, RetrySucceedsAfterCorrection) {
    // First attempt fails
    EXPECT_TRUE(recovery.can_retry("add"));
    recovery.record_retry("add");
    EXPECT_EQ(recovery.get_retry_count("add"), 1);

    // Still can retry
    EXPECT_TRUE(recovery.can_retry("add"));
}

// ============================================================================
// ER-006: Two retries fail -> ToolRetriesExhausted
// ============================================================================

TEST_F(ErrorRecoveryTest, RetriesExhausted) {
    recovery.record_retry("add");
    recovery.record_retry("add");

    EXPECT_FALSE(recovery.can_retry("add"));
    EXPECT_EQ(recovery.get_retry_count("add"), 2);
}

// ============================================================================
// ER-007: Multiple tool errors -> all tracked independently
// ============================================================================

TEST_F(ErrorRecoveryTest, MultipleToolsTracked) {
    recovery.record_retry("add");
    recovery.record_retry("greet");
    recovery.record_retry("add");

    EXPECT_FALSE(recovery.can_retry("add"));   // 2 retries
    EXPECT_TRUE(recovery.can_retry("greet"));   // 1 retry
    EXPECT_TRUE(recovery.can_retry("multiply")); // 0 retries
}

TEST_F(ErrorRecoveryTest, ResetClearsRetries) {
    recovery.record_retry("add");
    recovery.record_retry("add");
    EXPECT_FALSE(recovery.can_retry("add"));

    recovery.reset();
    EXPECT_TRUE(recovery.can_retry("add"));
    EXPECT_EQ(recovery.get_retry_count("add"), 0);
}

TEST_F(ErrorRecoveryTest, UnknownToolValidation) {
    ToolCall tc;
    tc.name = "nonexistent";
    tc.arguments = {};

    auto error = recovery.validate_args(tc, registry);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("not found"), std::string::npos);
}

// ============================================================================
// ToolCallParser Tests
// ============================================================================

class ToolCallParserTest : public ::testing::Test {};

// TA-003: Output contains tool call JSON -> detected and parsed
TEST_F(ToolCallParserTest, DetectsToolCall) {
    auto result = ToolCallParser::parse(TOOL_CALL_ADD);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 3);
    EXPECT_EQ(result.tool_call->arguments["b"], 4);
    EXPECT_FALSE(result.text_before.empty());
}

// TA-004: Output contains no tool call -> response returned directly
TEST_F(ToolCallParserTest, NoToolCall) {
    auto result = ToolCallParser::parse(PLAIN_TEXT);

    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, PLAIN_TEXT);
}

TEST_F(ToolCallParserTest, ToolCallOnly) {
    auto result = ToolCallParser::parse(TOOL_CALL_ONLY);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "greet");
    EXPECT_EQ(result.tool_call->arguments["name"], "Alice");
    EXPECT_TRUE(result.text_before.empty());
}

TEST_F(ToolCallParserTest, NestedJsonNotTool) {
    auto result = ToolCallParser::parse(NESTED_JSON_NOT_TOOL);

    // This JSON object doesn't have "name" + "arguments"
    EXPECT_FALSE(result.tool_call.has_value());
}

TEST_F(ToolCallParserTest, InvalidJsonSkipped) {
    auto result = ToolCallParser::parse(INVALID_JSON);
    EXPECT_FALSE(result.tool_call.has_value());
}

TEST_F(ToolCallParserTest, ToolCallWithId) {
    auto result = ToolCallParser::parse(TOOL_CALL_WITH_ID);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "search");
    EXPECT_EQ(result.tool_call->id, "call_123");
    EXPECT_EQ(result.tool_call->arguments["query"], "test");
}

TEST_F(ToolCallParserTest, ToolCallWithTrailingText) {
    auto result = ToolCallParser::parse(TOOL_CALL_WITH_TRAILING);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "multiply");
    EXPECT_NE(result.text_before.find("calculate"), std::string::npos);
}

TEST_F(ToolCallParserTest, EmptyOutput) {
    auto result = ToolCallParser::parse("");
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_TRUE(result.text_before.empty());
}
