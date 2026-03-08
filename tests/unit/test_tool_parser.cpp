/**
 * @file test_tool_parser.cpp
 * @brief Unit tests for heuristic and JSON-based tool-call parsing.
 */

#include <gtest/gtest.h>
#include "zoo/tools/parser.hpp"
#include "fixtures/sample_responses.hpp"

using namespace zoo::testing::responses;

TEST(ToolCallParserTest, DetectsToolCall) {
    auto result = zoo::tools::ToolCallParser::parse(TOOL_CALL_ADD);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 3);
    EXPECT_EQ(result.tool_call->arguments["b"], 4);
    EXPECT_FALSE(result.text_before.empty());
}

TEST(ToolCallParserTest, NoToolCall) {
    auto result = zoo::tools::ToolCallParser::parse(PLAIN_TEXT);
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, PLAIN_TEXT);
}

TEST(ToolCallParserTest, ToolCallOnly) {
    auto result = zoo::tools::ToolCallParser::parse(TOOL_CALL_ONLY);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "greet");
    EXPECT_EQ(result.tool_call->arguments["name"], "Alice");
    EXPECT_TRUE(result.text_before.empty());
}

TEST(ToolCallParserTest, NestedJsonNotTool) {
    auto result = zoo::tools::ToolCallParser::parse(NESTED_JSON_NOT_TOOL);
    EXPECT_FALSE(result.tool_call.has_value());
}

TEST(ToolCallParserTest, InvalidJson) {
    auto result = zoo::tools::ToolCallParser::parse(INVALID_JSON);
    EXPECT_FALSE(result.tool_call.has_value());
}

TEST(ToolCallParserTest, ToolCallWithId) {
    auto result = zoo::tools::ToolCallParser::parse(TOOL_CALL_WITH_ID);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->id, "call_123");
    EXPECT_EQ(result.tool_call->name, "search");
}

TEST(ToolCallParserTest, GeneratesIdWhenMissing) {
    auto result = zoo::tools::ToolCallParser::parse(TOOL_CALL_ONLY);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_FALSE(result.tool_call->id.empty());
    EXPECT_NE(result.tool_call->id.find("call_"), std::string::npos);
}

TEST(ToolCallParserTest, WithTrailingText) {
    auto result = zoo::tools::ToolCallParser::parse(TOOL_CALL_WITH_TRAILING);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "multiply");
}

TEST(ToolCallParserTest, EmptyInput) {
    auto result = zoo::tools::ToolCallParser::parse("");
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_TRUE(result.text_before.empty());
}
