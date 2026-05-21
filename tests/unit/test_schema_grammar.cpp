/**
 * @file test_schema_grammar.cpp
 * @brief Unit tests for GrammarBuilder::build_schema() (standalone schema grammars).
 */

#include "tools/grammar.hpp"
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
        {"color",
         ToolValueType::String,
         true,
         "",
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

    // Schema grammar should have the primitive rules but no tool-call sentinels
    EXPECT_EQ(schema_grammar.find("<tool_call>"), std::string::npos);
    EXPECT_NE(schema_grammar.find("string ::="), std::string::npos);
}

TEST(SchemaGrammarTest, IntegerRuleRejectsLeadingZeros) {
    std::vector<ToolParameter> params = {
        {"count", ToolValueType::Integer, true, "", {}},
    };
    auto grammar = GrammarBuilder::build_schema(params);
    // RFC 8259 §6: int = zero / ( digit1-9 *DIGIT ).  Permit "0" alone or a
    // non-zero leading digit; reject "0" followed by more digits.
    EXPECT_NE(grammar.find("integer ::= \"-\"? (\"0\" | [1-9] [0-9]*)"), std::string::npos);
}

TEST(SchemaGrammarTest, NumberRuleRejectsLeadingZeros) {
    std::vector<ToolParameter> params = {
        {"score", ToolValueType::Number, true, "", {}},
    };
    auto grammar = GrammarBuilder::build_schema(params);
    EXPECT_NE(grammar.find("number ::= \"-\"? (\"0\" | [1-9] [0-9]*)"), std::string::npos);
}

TEST(SchemaGrammarTest, StringRuleRejectsRawControlChars) {
    std::vector<ToolParameter> params = {
        {"text", ToolValueType::String, true, "", {}},
    };
    auto grammar = GrammarBuilder::build_schema(params);
    // The body character class must exclude raw control chars (U+0000..U+001F)
    // so that grammar-valid output is always parseable by nlohmann::json.
    EXPECT_NE(grammar.find("[^\"\\\\\\x00-\\x1F]"), std::string::npos);
}

TEST(SchemaGrammarTest, StringRuleAcceptsUnicodeEscape) {
    std::vector<ToolParameter> params = {
        {"text", ToolValueType::String, true, "", {}},
    };
    auto grammar = GrammarBuilder::build_schema(params);
    // RFC 8259 §7 \uXXXX escape must be in the alternation.
    EXPECT_NE(grammar.find("\"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]"),
              std::string::npos);
}

} // namespace
