/**
 * @file test_schema_grammar.cpp
 * @brief Unit tests for GrammarBuilder::build_schema() (standalone schema grammars).
 */

#include "zoo/internal/tools/grammar.hpp"
#include <gtest/gtest.h>

namespace {

using zoo::tools::GrammarBuilder;
using zoo::tools::ToolParameter;
using zoo::tools::ToolValueType;

TEST(SchemaGrammarTest, EmptyParametersProducesMinimalGrammar) {
    auto grammar = GrammarBuilder::build_schema({});

    EXPECT_NE(grammar.find("root ::="), std::string::npos);
    EXPECT_NE(grammar.find("ws ::="), std::string::npos);
    // Should produce a grammar for empty object
    EXPECT_EQ(grammar.find("<tool_call>"), std::string::npos);
    EXPECT_EQ(grammar.find("</tool_call>"), std::string::npos);
}

TEST(SchemaGrammarTest, NoToolCallSentinelsInOutput) {
    std::vector<ToolParameter> params = {
        {"name", ToolValueType::String, true, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);

    EXPECT_EQ(grammar.find("<tool_call>"), std::string::npos);
    EXPECT_EQ(grammar.find("</tool_call>"), std::string::npos);
    EXPECT_EQ(grammar.find("tool-call"), std::string::npos);
}

TEST(SchemaGrammarTest, SingleRequiredStringProperty) {
    std::vector<ToolParameter> params = {
        {"name", ToolValueType::String, true, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);

    EXPECT_NE(grammar.find("root ::="), std::string::npos);
    EXPECT_NE(grammar.find("schema-0-args"), std::string::npos);
    EXPECT_NE(grammar.find("schema-0-param-0"), std::string::npos);
    EXPECT_NE(grammar.find("\\\"name\\\""), std::string::npos);
    EXPECT_NE(grammar.find("string ::="), std::string::npos);
}

TEST(SchemaGrammarTest, MultiplePropertiesMixedRequiredOptional) {
    std::vector<ToolParameter> params = {
        {"city", ToolValueType::String, true, "", {}},
        {"age", ToolValueType::Integer, true, "", {}},
        {"nickname", ToolValueType::String, false, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);

    EXPECT_NE(grammar.find("schema-0-param-0"), std::string::npos);
    EXPECT_NE(grammar.find("schema-0-param-1"), std::string::npos);
    EXPECT_NE(grammar.find("schema-0-param-2"), std::string::npos);
    EXPECT_NE(grammar.find("\\\"city\\\""), std::string::npos);
    EXPECT_NE(grammar.find("\\\"age\\\""), std::string::npos);
    EXPECT_NE(grammar.find("\\\"nickname\\\""), std::string::npos);
    // Optional rules for the third parameter
    EXPECT_NE(grammar.find("schema-0-cont-2"), std::string::npos);
}

TEST(SchemaGrammarTest, IntegerType) {
    std::vector<ToolParameter> params = {
        {"count", ToolValueType::Integer, true, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);
    EXPECT_NE(grammar.find("integer ::="), std::string::npos);
}

TEST(SchemaGrammarTest, NumberType) {
    std::vector<ToolParameter> params = {
        {"score", ToolValueType::Number, true, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);
    EXPECT_NE(grammar.find("number ::="), std::string::npos);
}

TEST(SchemaGrammarTest, BooleanType) {
    std::vector<ToolParameter> params = {
        {"active", ToolValueType::Boolean, true, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);
    EXPECT_NE(grammar.find("boolean ::="), std::string::npos);
}

TEST(SchemaGrammarTest, EnumConstrainedValues) {
    std::vector<ToolParameter> params = {
        {"color", ToolValueType::String, true, "",
         {nlohmann::json("red"), nlohmann::json("green"), nlohmann::json("blue")}},
    };

    auto grammar = GrammarBuilder::build_schema(params);

    EXPECT_NE(grammar.find("schema-0-enum-0"), std::string::npos);
    EXPECT_NE(grammar.find("\\\"red\\\""), std::string::npos);
    EXPECT_NE(grammar.find("\\\"green\\\""), std::string::npos);
    EXPECT_NE(grammar.find("\\\"blue\\\""), std::string::npos);
}

TEST(SchemaGrammarTest, AllOptionalProperties) {
    std::vector<ToolParameter> params = {
        {"a", ToolValueType::String, false, "", {}},
        {"b", ToolValueType::Integer, false, "", {}},
    };

    auto grammar = GrammarBuilder::build_schema(params);

    // Start rule should be referenced (all optional)
    EXPECT_NE(grammar.find("schema-0-start-0"), std::string::npos);
}

TEST(SchemaGrammarTest, DiffersFromToolGrammar) {
    std::vector<ToolParameter> params = {
        {"query", ToolValueType::String, true, "", {}},
    };

    auto schema_grammar = GrammarBuilder::build_schema(params);

    // Build an equivalent tool grammar for comparison
    zoo::tools::ToolMetadata tool_meta;
    tool_meta.name = "search";
    tool_meta.description = "Search";
    tool_meta.parameters = params;
    auto tool_grammar = GrammarBuilder::build({tool_meta});

    // Schema grammar should NOT have tool-call sentinels
    EXPECT_EQ(schema_grammar.find("<tool_call>"), std::string::npos);
    // Tool grammar SHOULD have tool-call sentinels
    EXPECT_NE(tool_grammar.find("<tool_call>"), std::string::npos);

    // Both should have the primitive rules
    EXPECT_NE(schema_grammar.find("string ::="), std::string::npos);
    EXPECT_NE(tool_grammar.find("string ::="), std::string::npos);
}

} // namespace
