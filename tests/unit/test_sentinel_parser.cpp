#include <gtest/gtest.h>
#include "zoo/tools/parser.hpp"

TEST(SentinelParserTest, BasicSentinelExtraction) {
    std::string output =
        R"(I'll add those numbers for you.
<tool_call>{"name": "add", "arguments": {"a": 3, "b": 4}}</tool_call>)";

    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 3);
    EXPECT_EQ(result.tool_call->arguments["b"], 4);
    EXPECT_EQ(result.text_before, "I'll add those numbers for you.\n");
}

TEST(SentinelParserTest, NoSentinel) {
    std::string output = "Just a normal response with no tool calls.";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, output);
}

TEST(SentinelParserTest, SentinelOnly) {
    std::string output =
        R"(<tool_call>{"name": "get_time", "arguments": {}}</tool_call>)";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "get_time");
    EXPECT_TRUE(result.text_before.empty());
}

TEST(SentinelParserTest, IncompleteSentinel) {
    std::string output =
        R"(Let me check... <tool_call>{"name": "add", "arguments": {"a": 3, "b": 4})";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, output);
}

TEST(SentinelParserTest, InvalidJsonInSentinel) {
    std::string output =
        R"(<tool_call>{"name": "add", "arguments": {"a": 3, "b":}</tool_call>)";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, output);
}

TEST(SentinelParserTest, WhitespaceAroundJson) {
    std::string output = "<tool_call>\n  {\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}\n</tool_call>";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
}

TEST(SentinelParserTest, SentinelWithId) {
    std::string output =
        R"(<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}, "id": "call_42"}</tool_call>)";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->id, "call_42");
}

TEST(SentinelParserTest, GeneratesIdWhenMissing) {
    std::string output =
        R"(<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>)";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_FALSE(result.tool_call->id.empty());
    EXPECT_NE(result.tool_call->id.find("call_"), std::string::npos);
}

TEST(SentinelParserTest, EmptyInput) {
    auto result = zoo::tools::ToolCallParser::parse_sentinel("");
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_TRUE(result.text_before.empty());
}

TEST(SentinelParserTest, ChainOfThoughtBeforeSentinel) {
    std::string output =
        R"(The user wants to add 22 and 57, then multiply by 1.6.
I'll start with the addition.
<tool_call>{"name": "add", "arguments": {"a": 22, "b": 57}}</tool_call>)";

    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    ASSERT_TRUE(result.tool_call.has_value());
    EXPECT_EQ(result.tool_call->name, "add");
    EXPECT_EQ(result.tool_call->arguments["a"], 22);
    EXPECT_NE(result.text_before.find("addition"), std::string::npos);
}

TEST(SentinelParserTest, MissingNameField) {
    std::string output =
        R"(<tool_call>{"arguments": {"a": 1}}</tool_call>)";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    EXPECT_FALSE(result.tool_call.has_value());
}

TEST(SentinelParserTest, CodeBlockWithBracesNotConfused) {
    std::string output = "Here's some C++ code:\n```cpp\nint main() {\n  return 0;\n}\n```";
    auto result = zoo::tools::ToolCallParser::parse_sentinel(output);
    EXPECT_FALSE(result.tool_call.has_value());
    EXPECT_EQ(result.text_before, output);
}
