/**
 * @file test_error_recovery.cpp
 * @brief Unit tests for tool argument validation.
 */

#include "fixtures/tool_definitions.hpp"
#include "zoo/tools/validation.hpp"
#include <gtest/gtest.h>

using namespace zoo::testing::tools;

/// Shared fixture that pre-registers common tools for validation tests.
class ToolArgumentsValidatorTest : public ::testing::Test {
  protected:
    zoo::tools::ToolRegistry registry;
    zoo::tools::ToolArgumentsValidator validator;

    void SetUp() override {
        ASSERT_TRUE(registry.register_tool("add", "Add two integers", {"a", "b"}, add).has_value());
        ASSERT_TRUE(registry.register_tool("greet", "Greet someone", {"name"}, greet).has_value());
        ASSERT_TRUE(registry.register_tool("multiply", "Multiply doubles", {"a", "b"}, multiply)
                        .has_value());

        nlohmann::json schema = {
            {"type", "object"},
            {"properties",
             {{"unit",
               {{"type", "string"}, {"enum", nlohmann::json::array({"celsius", "fahrenheit"})}}},
              {"days", {{"type", "integer"}}}}},
            {"required", nlohmann::json::array({"unit"})},
            {"additionalProperties", false}};

        ASSERT_TRUE(registry
                        .register_tool("forecast", "Fetch forecast", schema,
                                       [](const nlohmann::json&) -> zoo::Expected<nlohmann::json> {
                                           return nlohmann::json::object();
                                       })
                        .has_value());
    }
};

TEST_F(ToolArgumentsValidatorTest, ValidArgsPass) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}, {"b", 4}};
    EXPECT_TRUE(validator.validate(tc, registry).has_value());
}

TEST_F(ToolArgumentsValidatorTest, ValidStringArgs) {
    zoo::tools::ToolCall tc;
    tc.name = "greet";
    tc.arguments = {{"name", "Alice"}};
    EXPECT_TRUE(validator.validate(tc, registry).has_value());
}

TEST_F(ToolArgumentsValidatorTest, MissingRequiredArgFails) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", 3}};
    auto result = validator.validate(tc, registry);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolValidationFailed);
    EXPECT_NE(result.error().message.find("Missing"), std::string::npos);
}

TEST_F(ToolArgumentsValidatorTest, WrongArgTypeFails) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = {{"a", "three"}, {"b", 4}};
    auto result = validator.validate(tc, registry);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolValidationFailed);
    EXPECT_NE(result.error().message.find("wrong type"), std::string::npos);
}

TEST_F(ToolArgumentsValidatorTest, UnknownArgumentFails) {
    zoo::tools::ToolCall tc;
    tc.name = "greet";
    tc.arguments = {{"name", "Alice"}, {"salutation", "Hi"}};
    auto result = validator.validate(tc, registry);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolValidationFailed);
    EXPECT_NE(result.error().message.find("Unexpected argument"), std::string::npos);
}

TEST_F(ToolArgumentsValidatorTest, UnknownToolFails) {
    zoo::tools::ToolCall tc;
    tc.name = "nonexistent";
    tc.arguments = {};
    auto result = validator.validate(tc, registry);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolNotFound);
}

TEST_F(ToolArgumentsValidatorTest, EnumArgumentMustMatchRegisteredValues) {
    zoo::tools::ToolCall tc;
    tc.name = "forecast";
    tc.arguments = {{"unit", "kelvin"}};
    auto result = validator.validate(tc, registry);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolValidationFailed);
    EXPECT_NE(result.error().message.find("enum"), std::string::npos);
}

TEST_F(ToolArgumentsValidatorTest, OptionalArgumentMayBeOmitted) {
    zoo::tools::ToolCall tc;
    tc.name = "forecast";
    tc.arguments = {{"unit", "celsius"}};
    EXPECT_TRUE(validator.validate(tc, registry).has_value());
}

TEST_F(ToolArgumentsValidatorTest, NonObjectArgumentsFail) {
    zoo::tools::ToolCall tc;
    tc.name = "add";
    tc.arguments = nlohmann::json::array({1, 2});
    auto result = validator.validate(tc, registry);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolValidationFailed);
    EXPECT_NE(result.error().message.find("JSON object"), std::string::npos);
}

TEST_F(ToolArgumentsValidatorTest, ValidEnumArgumentPasses) {
    zoo::tools::ToolCall tc;
    tc.name = "forecast";
    tc.arguments = {{"unit", "fahrenheit"}, {"days", 3}};
    EXPECT_TRUE(validator.validate(tc, registry).has_value());
}
