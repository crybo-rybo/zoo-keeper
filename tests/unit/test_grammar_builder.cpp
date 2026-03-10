/**
 * @file test_grammar_builder.cpp
 * @brief Unit tests for grammar generation from registered tool metadata.
 */

#include "zoo/internal/tools/grammar.hpp"
#include "zoo/tools/registry.hpp"
#include <gtest/gtest.h>

static int add(int a, int b) {
    return a + b;
}
static std::string greet(std::string name) {
    return "Hi " + name;
}
static double scale(double x, double factor) {
    return x * factor;
}
static bool negate(bool val) {
    return !val;
}
static std::string get_time() {
    return "now";
}

/// Shared fixture that exposes a registry for grammar construction tests.
class GrammarBuilderTest : public ::testing::Test {
  protected:
    zoo::tools::ToolRegistry registry;
};

TEST_F(GrammarBuilderTest, EmptyRegistryReturnsEmpty) {
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_TRUE(grammar.empty());
}

TEST_F(GrammarBuilderTest, SingleToolProducesValidGrammar) {
    ASSERT_TRUE(registry.register_tool("add", "Add two numbers", {"a", "b"}, add).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());

    EXPECT_NE(grammar.find("root ::="), std::string::npos);
    EXPECT_NE(grammar.find("<tool_call>"), std::string::npos);
    EXPECT_NE(grammar.find("</tool_call>"), std::string::npos);
    EXPECT_NE(grammar.find("tool-call ::= tool-0"), std::string::npos);
    EXPECT_NE(grammar.find("\\\"add\\\""), std::string::npos);
}

TEST_F(GrammarBuilderTest, MultipleToolsRespectRegistrationOrder) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet", {"name"}, greet).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add", {"a", "b"}, add).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    auto tool_call_pos = grammar.find("tool-call ::= tool-0 | tool-1");
    auto greet_pos = grammar.find("\\\"greet\\\"");
    auto add_pos = grammar.find("\\\"add\\\"");

    EXPECT_NE(tool_call_pos, std::string::npos);
    ASSERT_NE(greet_pos, std::string::npos);
    ASSERT_NE(add_pos, std::string::npos);
    EXPECT_LT(greet_pos, add_pos);
}

TEST_F(GrammarBuilderTest, IntegerArgsReferenceIntegerRule) {
    ASSERT_TRUE(registry.register_tool("add", "Add", {"a", "b"}, add).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("integer ::="), std::string::npos);
    EXPECT_NE(grammar.find("tool-0-param-0"), std::string::npos);
}

TEST_F(GrammarBuilderTest, NumberArgsReferenceNumberRule) {
    ASSERT_TRUE(registry.register_tool("scale", "Scale", {"x", "factor"}, scale).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("number ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, StringArgsReferenceStringRule) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet", {"name"}, greet).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("string ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, BooleanArgsReferenceBooleanRule) {
    ASSERT_TRUE(registry.register_tool("negate", "Negate", {"val"}, negate).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("boolean ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, ZeroArityToolHasNoArgsRule) {
    ASSERT_TRUE(registry.register_tool("get_time", "Get time", {}, get_time).has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("\\\"get_time\\\""), std::string::npos);
    EXPECT_EQ(grammar.find("tool-0-args ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, ManualSchemaGeneratesOptionalRules) {
    nlohmann::json schema = {{"type", "object"},
                             {"properties",
                              {{"query", {{"type", "string"}}},
                               {"limit", {{"type", "integer"}}}}},
                             {"required", nlohmann::json::array({"query"})},
                             {"additionalProperties", false}};

    ASSERT_TRUE(registry.register_tool(
                           "search", "Search", schema,
                           [](const nlohmann::json&) -> zoo::Expected<nlohmann::json> {
                               return nlohmann::json::object();
                           })
                    .has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("tool-0-cont-1"), std::string::npos);
    EXPECT_NE(grammar.find("\\\"limit\\\""), std::string::npos);
}

TEST_F(GrammarBuilderTest, ManualSchemaGeneratesEnumRules) {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties",
         {{"unit", {{"type", "string"}, {"enum", nlohmann::json::array({"celsius", "fahrenheit"})}}}}},
        {"required", nlohmann::json::array({"unit"})},
        {"additionalProperties", false}};

    ASSERT_TRUE(registry.register_tool(
                           "weather", "Weather", schema,
                           [](const nlohmann::json&) -> zoo::Expected<nlohmann::json> {
                               return nlohmann::json::object();
                           })
                    .has_value());

    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_tool_metadata());
    EXPECT_NE(grammar.find("tool-0-enum-0"), std::string::npos);
    EXPECT_NE(grammar.find("\\\"celsius\\\""), std::string::npos);
    EXPECT_NE(grammar.find("\\\"fahrenheit\\\""), std::string::npos);
}
