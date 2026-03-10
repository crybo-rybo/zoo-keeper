/**
 * @file test_grammar_builder.cpp
 * @brief Unit tests for grammar generation from registered tool schemas.
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
    auto schemas = registry.get_all_schemas();
    auto grammar = zoo::tools::GrammarBuilder::build(schemas);
    EXPECT_TRUE(grammar.empty());
}

TEST_F(GrammarBuilderTest, SingleToolProducesValidGrammar) {
    registry.register_tool("add", "Add two numbers", {"a", "b"}, add);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("root ::="), std::string::npos);
    EXPECT_NE(grammar.find("<tool_call>"), std::string::npos);
    EXPECT_NE(grammar.find("</tool_call>"), std::string::npos);
    EXPECT_NE(grammar.find("tool-call ::= tool-add"), std::string::npos);
    EXPECT_NE(grammar.find("tool-add ::="), std::string::npos);
    EXPECT_NE(grammar.find("\\\"add\\\""), std::string::npos);
}

TEST_F(GrammarBuilderTest, MultipleToolsCreateAlternatives) {
    registry.register_tool("add", "Add", {"a", "b"}, add);
    registry.register_tool("greet", "Greet", {"name"}, greet);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    // Both tools must appear as alternatives (order depends on registry internals)
    EXPECT_NE(grammar.find("tool-add"), std::string::npos);
    EXPECT_NE(grammar.find("tool-greet"), std::string::npos);
    EXPECT_NE(grammar.find("tool-call ::="), std::string::npos);
    EXPECT_NE(grammar.find(" | "), std::string::npos);
}

TEST_F(GrammarBuilderTest, IntegerArgsReferenceIntegerRule) {
    registry.register_tool("add", "Add", {"a", "b"}, add);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("add-args ::="), std::string::npos);
    EXPECT_NE(grammar.find("\\\"a\\\"\" ws \":\" ws integer"), std::string::npos);
    EXPECT_NE(grammar.find("integer ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, NumberArgsReferenceNumberRule) {
    registry.register_tool("scale", "Scale", {"x", "factor"}, scale);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("\\\"x\\\"\" ws \":\" ws number"), std::string::npos);
    EXPECT_NE(grammar.find("number ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, StringArgsReferenceStringRule) {
    registry.register_tool("greet", "Greet", {"name"}, greet);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("\\\"name\\\"\" ws \":\" ws string"), std::string::npos);
    EXPECT_NE(grammar.find("string ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, BooleanArgsReferenceBooleanRule) {
    registry.register_tool("negate", "Negate", {"val"}, negate);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("\\\"val\\\"\" ws \":\" ws boolean"), std::string::npos);
    EXPECT_NE(grammar.find("boolean ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, ZeroArityToolHasNoArgsRule) {
    registry.register_tool("get_time", "Get time", {}, get_time);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("tool-get-time ::="), std::string::npos);
    // Should NOT have a get-time-args rule
    EXPECT_EQ(grammar.find("get-time-args ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, ContainsPrimitiveRules) {
    registry.register_tool("add", "Add", {"a", "b"}, add);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("integer ::="), std::string::npos);
    EXPECT_NE(grammar.find("number ::="), std::string::npos);
    EXPECT_NE(grammar.find("string ::="), std::string::npos);
    EXPECT_NE(grammar.find("boolean ::="), std::string::npos);
    EXPECT_NE(grammar.find("ws ::="), std::string::npos);
}

TEST_F(GrammarBuilderTest, ToolNameIsLiteralInGrammar) {
    registry.register_tool("multiply", "Multiply", {"a", "b"}, scale);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    // The tool name must appear as a literal string in the grammar
    EXPECT_NE(grammar.find("\\\"multiply\\\""), std::string::npos);
}

TEST_F(GrammarBuilderTest, ArgumentNamesAreLiterals) {
    registry.register_tool("add", "Add", {"left", "right"}, add);
    auto grammar = zoo::tools::GrammarBuilder::build(registry.get_all_schemas());

    EXPECT_NE(grammar.find("\\\"left\\\""), std::string::npos);
    EXPECT_NE(grammar.find("\\\"right\\\""), std::string::npos);
}
