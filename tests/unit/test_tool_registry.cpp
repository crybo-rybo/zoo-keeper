/**
 * @file test_tool_registry.cpp
 * @brief Unit tests for tool registration, schema generation, and invocation.
 */

#include <gtest/gtest.h>
#include "zoo/tools/registry.hpp"
#include "fixtures/tool_definitions.hpp"

using json = nlohmann::json;
using namespace zoo::testing::tools;

/// Shared fixture that provides a fresh tool registry for each test.
class ToolRegistryTest : public ::testing::Test {
protected:
    zoo::tools::ToolRegistry registry;
};

TEST_F(ToolRegistryTest, RegisterIntParams) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);
    EXPECT_TRUE(registry.has_tool("add"));
    EXPECT_EQ(registry.size(), 1);

    auto schema = registry.get_tool_schema("add");
    EXPECT_EQ(schema["function"]["name"], "add");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["a"]["type"], "integer");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["b"]["type"], "integer");
}

TEST_F(ToolRegistryTest, RegisterStringParams) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);
    auto schema = registry.get_tool_schema("greet");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["name"]["type"], "string");
}

TEST_F(ToolRegistryTest, RegisterDoubleParams) {
    (void)registry.register_tool("multiply", "Multiply doubles", {"a", "b"}, multiply);
    auto schema = registry.get_tool_schema("multiply");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["a"]["type"], "number");
}

TEST_F(ToolRegistryTest, RegisterBoolParams) {
    (void)registry.register_tool("is_positive", "Check positive", {"n"}, is_positive);
    auto schema = registry.get_tool_schema("is_positive");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["n"]["type"], "integer");
}

TEST_F(ToolRegistryTest, RegisterZeroArity) {
    (void)registry.register_tool("get_time", "Get time", {}, get_time);
    EXPECT_TRUE(registry.has_tool("get_time"));
    auto result = registry.invoke("get_time", json::object());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "2024-01-01T00:00:00Z");
}

TEST_F(ToolRegistryTest, InvokeSuccess) {
    (void)registry.register_tool("add", "Add", {"a", "b"}, add);
    auto result = registry.invoke("add", {{"a", 3}, {"b", 4}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 7);
}

TEST_F(ToolRegistryTest, InvokeNotFound) {
    auto result = registry.invoke("nonexistent", {});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolNotFound);
}

TEST_F(ToolRegistryTest, ArityMismatch) {
    auto result = registry.register_tool("add", "Add", {"a"}, add); // add needs 2
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSignature);
}

TEST_F(ToolRegistryTest, GetAllSchemas) {
    (void)registry.register_tool("add", "Add", {"a", "b"}, add);
    (void)registry.register_tool("greet", "Greet", {"name"}, greet);
    auto schemas = registry.get_all_schemas();
    EXPECT_EQ(schemas.size(), 2);
}

TEST_F(ToolRegistryTest, GetToolNames) {
    (void)registry.register_tool("add", "Add", {"a", "b"}, add);
    (void)registry.register_tool("greet", "Greet", {"name"}, greet);
    auto names = registry.get_tool_names();
    EXPECT_EQ(names.size(), 2);
}

TEST_F(ToolRegistryTest, HasToolFalse) {
    EXPECT_FALSE(registry.has_tool("nonexistent"));
}

TEST_F(ToolRegistryTest, GetParametersSchema) {
    (void)registry.register_tool("add", "Add", {"a", "b"}, add);
    auto schema = registry.get_parameters_schema("add");
    ASSERT_TRUE(schema.has_value());
    EXPECT_TRUE(schema->contains("required"));

    auto missing = registry.get_parameters_schema("nonexistent");
    EXPECT_FALSE(missing.has_value());
}

TEST_F(ToolRegistryTest, LambdaRegistration) {
    auto result = registry.register_tool("double_it", "Double a number", {"x"},
        [](int x) { return x * 2; });
    EXPECT_TRUE(result.has_value());

    auto invoke_result = registry.invoke("double_it", {{"x", 5}});
    ASSERT_TRUE(invoke_result.has_value());
    EXPECT_EQ((*invoke_result)["result"], 10);
}

TEST_F(ToolRegistryTest, OverwriteExisting) {
    (void)registry.register_tool("add", "Add v1", {"a", "b"}, add);
    (void)registry.register_tool("add", "Add v2", {"a", "b"}, add);
    EXPECT_EQ(registry.size(), 1);
    auto schema = registry.get_tool_schema("add");
    EXPECT_EQ(schema["function"]["description"], "Add v2");
}
