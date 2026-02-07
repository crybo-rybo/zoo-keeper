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

// ============================================================================
// Additional ErrorRecovery coverage
// ============================================================================

TEST_F(ErrorRecoveryTest, MaxRetriesAccessor) {
    EXPECT_EQ(recovery.max_retries(), 2);

    ErrorRecovery custom_recovery(5);
    EXPECT_EQ(custom_recovery.max_retries(), 5);
}

TEST_F(ErrorRecoveryTest, DefaultMaxRetries) {
    ErrorRecovery default_recovery;
    EXPECT_EQ(default_recovery.max_retries(), 2);
}

TEST_F(ErrorRecoveryTest, ZeroMaxRetriesExhaustsAfterFirstRecord) {
    ErrorRecovery zero_recovery(0);
    // First check: no retries recorded yet, so it can try
    EXPECT_TRUE(zero_recovery.can_retry("any_tool"));
    // After recording one retry, immediately exhausted
    zero_recovery.record_retry("any_tool");
    EXPECT_FALSE(zero_recovery.can_retry("any_tool"));
}

TEST_F(ErrorRecoveryTest, ExtraArgsStillValid) {
    ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}, {"b", 4}, {"extra", "ignored"}};

    auto error = recovery.validate_args(tc, registry);
    EXPECT_TRUE(error.empty());
}

TEST_F(ErrorRecoveryTest, NumberTypeAcceptsInt) {
    // "multiply" has double params -> schema type "number"
    // Passing integer values should be accepted since int is_number()
    ToolCall tc;
    tc.name = "multiply";
    tc.arguments = {{"a", 3}, {"b", 4}};

    auto error = recovery.validate_args(tc, registry);
    EXPECT_TRUE(error.empty());
}

TEST_F(ErrorRecoveryTest, SchemaLacksProperties) {
    // Register tool with minimal schema (no properties key)
    ToolRegistry minimal_reg;
    nlohmann::json minimal_schema = {{"type", "object"}};
    ToolHandler bare_handler = [](const nlohmann::json&) -> Expected<nlohmann::json> {
        return nlohmann::json{{"result", "ok"}};
    };
    minimal_reg.register_tool("bare", "Bare tool", std::move(minimal_schema),
        std::move(bare_handler));

    ToolCall tc;
    tc.name = "bare";
    tc.arguments = {{"any", "value"}};

    auto error = recovery.validate_args(tc, minimal_reg);
    EXPECT_TRUE(error.empty());  // Should pass — no properties to check
}

TEST_F(ErrorRecoveryTest, PropertyLacksTypeField) {
    // Register tool with property that has no "type" key
    ToolRegistry special_reg;
    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {{"x", {{"description", "something"}}}}},
        {"required", nlohmann::json::array()}
    };
    ToolHandler special_handler = [](const nlohmann::json&) -> Expected<nlohmann::json> {
        return nlohmann::json{{"result", "ok"}};
    };
    special_reg.register_tool("special", "Special tool", std::move(schema),
        std::move(special_handler));

    ToolCall tc;
    tc.name = "special";
    tc.arguments = {{"x", 42}};

    auto error = recovery.validate_args(tc, special_reg);
    EXPECT_TRUE(error.empty());  // Should skip type check
}

TEST_F(ErrorRecoveryTest, GetRetryCountForUnknownTool) {
    EXPECT_EQ(recovery.get_retry_count("never_seen"), 0);
}

// ============================================================================
// Additional ToolCallParser coverage
// ============================================================================

TEST_F(ToolCallParserTest, DeeplyNestedJson) {
    std::string input = R"({"name": "search", "arguments": {"query": {"nested": {"deep": true}}}})";
    auto result = ToolCallParser::parse(input);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "search");
    EXPECT_TRUE(result.tool_call->arguments["query"]["nested"]["deep"].get<bool>());
}

TEST_F(ToolCallParserTest, EscapedQuotesInStrings) {
    std::string input = R"({"name": "greet", "arguments": {"name": "O\"Brien"}})";
    auto result = ToolCallParser::parse(input);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "greet");
    EXPECT_EQ(result.tool_call->arguments["name"], "O\"Brien");
}

TEST_F(ToolCallParserTest, MultipleJsonObjectsFirstDetected) {
    std::string input = R"({"name": "add", "arguments": {"a": 1, "b": 2}} and {"name": "greet", "arguments": {"name": "X"}})";
    auto result = ToolCallParser::parse(input);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");  // First one wins
}

TEST_F(ToolCallParserTest, WhitespaceOnlyInput) {
    auto result = ToolCallParser::parse("   \n\t  ");
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, "   \n\t  ");
}

TEST_F(ToolCallParserTest, AutoGeneratedIdsAreSequential) {
    auto r1 = ToolCallParser::parse(R"({"name": "add", "arguments": {"a": 1, "b": 2}})");
    auto r2 = ToolCallParser::parse(R"({"name": "add", "arguments": {"a": 3, "b": 4}})");

    ASSERT_TRUE(r1.tool_call.has_value());
    ASSERT_TRUE(r2.tool_call.has_value());

    // Both should have auto-generated IDs starting with "call_"
    EXPECT_EQ(r1.tool_call->id.substr(0, 5), "call_");
    EXPECT_EQ(r2.tool_call->id.substr(0, 5), "call_");
    EXPECT_NE(r1.tool_call->id, r2.tool_call->id);
}

TEST_F(ToolCallParserTest, UnbalancedBracesSkipped) {
    std::string input = R"({{"name": "add"} plain text after)";
    auto result = ToolCallParser::parse(input);
    // Unbalanced braces → no valid tool call
    EXPECT_FALSE(result.tool_call.has_value());
}

TEST_F(ToolCallParserTest, JsonWithBracesInsideStringValues) {
    std::string input = R"({"name": "echo", "arguments": {"text": "value with { braces }"}})";
    auto result = ToolCallParser::parse(input);

    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "echo");
    EXPECT_EQ(result.tool_call->arguments["text"], "value with { braces }");
}
