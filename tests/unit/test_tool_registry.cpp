/**
 * @file test_tool_registry.cpp
 * @brief Unit tests for tool registration, schema normalization, and invocation.
 */

#include "fixtures/tool_definitions.hpp"
#include "zoo/tools/registry.hpp"
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <thread>

using json = nlohmann::json;
using namespace zoo::testing::tools;

/// Shared fixture that provides a fresh tool registry for each test.
class ToolRegistryTest : public ::testing::Test {
  protected:
    zoo::tools::ToolRegistry registry;
};

TEST_F(ToolRegistryTest, RegisterIntParams) {
    ASSERT_TRUE(registry.register_tool("add", "Add two integers", {"a", "b"}, add).has_value());

    auto schema = registry.get_tool_schema("add");
    ASSERT_TRUE(schema.is_object());
    EXPECT_EQ(schema["function"]["name"], "add");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["a"]["type"], "integer");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["b"]["type"], "integer");
    EXPECT_EQ(schema["function"]["parameters"]["required"], json::array({"a", "b"}));
    EXPECT_EQ(schema["function"]["parameters"]["additionalProperties"], false);
}

TEST_F(ToolRegistryTest, RegisterStringParams) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet someone", {"name"}, greet).has_value());

    auto schema = registry.get_tool_schema("greet");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["name"]["type"], "string");
}

TEST_F(ToolRegistryTest, RegisterZeroArity) {
    ASSERT_TRUE(registry.register_tool("get_time", "Get time", {}, get_time).has_value());

    auto schema = registry.get_tool_schema("get_time");
    EXPECT_TRUE(schema["function"]["parameters"]["properties"].empty());
    EXPECT_TRUE(schema["function"]["parameters"]["required"].empty());

    auto result = registry.invoke("get_time", json::object());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "2024-01-01T00:00:00Z");
}

TEST_F(ToolRegistryTest, ManualSchemaRegistrationNormalizesOptionalAndEnumFields) {
    json schema = {{"type", "object"},
                   {"properties",
                    {{"query", {{"type", "string"}, {"description", "Search query"}}},
                     {"limit", {{"type", "integer"}, {"enum", json::array({5, 10, 20})}}}}},
                   {"required", json::array({"query"})},
                   {"additionalProperties", false}};

    zoo::tools::ToolHandler handler = [](const json& args) -> zoo::Expected<json> {
        return json{{"query", args.at("query")}, {"limit", args.value("limit", 10)}};
    };

    auto result = registry.register_tool("search", "Search documents", schema, std::move(handler));
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    auto metadata = registry.get_tool_metadata("search");
    ASSERT_TRUE(metadata.has_value());
    ASSERT_EQ(metadata->parameters.size(), 2u);

    EXPECT_EQ(metadata->parameters[0].name, "query");
    EXPECT_TRUE(metadata->parameters[0].required);
    EXPECT_EQ(metadata->parameters[1].name, "limit");
    EXPECT_FALSE(metadata->parameters[1].required);
    EXPECT_EQ(metadata->parameters[1].enum_values,
              json::array({5, 10, 20}).get<std::vector<json>>());
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsNestedObjects) {
    json schema = {
        {"type", "object"},
        {"properties",
         {{"query",
           {{"type", "object"},
            {"properties", {{"term", {{"type", "string"}}}}},
            {"required", json::array({"term"})}}}}},
        {"required", json::array({"query"})},
    };

    auto result =
        registry.register_tool("search_documents", "Search with nested query", schema,
                               [](const json&) -> zoo::Expected<json> { return json::object(); });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsUnsupportedKeywords) {
    json schema = {{"type", "object"},
                   {"properties", {{"limit", {{"type", "integer"}, {"minimum", 1}}}}}};

    auto result =
        registry.register_tool("bounded_limit", "Schema with unsupported bounds", schema,
                               [](const json&) -> zoo::Expected<json> { return json::object(); });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("minimum"), std::string::npos);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsRefKeyword) {
    json schema = {{"type", "object"},
                   {"properties", {{"data", {{"$ref", "#/definitions/Data"}}}}},
                   {"required", json::array({"data"})}};

    auto result =
        registry.register_tool("ref_tool", "Schema with $ref", schema,
                               [](const json&) -> zoo::Expected<json> { return json::object(); });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsArrayType) {
    json schema = {
        {"type", "object"},
        {"properties", {{"items", {{"type", "array"}, {"items", {{"type", "string"}}}}}}},
        {"required", json::array({"items"})}};

    auto result =
        registry.register_tool("array_tool", "Schema with array", schema,
                               [](const json&) -> zoo::Expected<json> { return json::object(); });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST_F(ToolRegistryTest, InvokeSuccess) {
    ASSERT_TRUE(registry.register_tool("add", "Add", {"a", "b"}, add).has_value());

    auto result = registry.invoke("add", {{"a", 3}, {"b", 4}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 7);
}

TEST_F(ToolRegistryTest, InvokeNotFound) {
    auto result = registry.invoke("nonexistent", {});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::ToolNotFound);
}

TEST_F(ToolRegistryTest, ArityMismatch) {
    auto result = registry.register_tool("add", "Add", {"a"}, add);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSignature);
}

TEST_F(ToolRegistryTest, GetAllSchemasUsesRegistrationOrder) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet", {"name"}, greet).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add", {"a", "b"}, add).has_value());

    auto schemas = registry.get_all_schemas();
    ASSERT_EQ(schemas.size(), 2u);
    EXPECT_EQ(schemas[0]["function"]["name"], "greet");
    EXPECT_EQ(schemas[1]["function"]["name"], "add");
}

TEST_F(ToolRegistryTest, GetToolNamesUsesRegistrationOrder) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet", {"name"}, greet).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add", {"a", "b"}, add).has_value());

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"greet", "add"}));
}

TEST_F(ToolRegistryTest, GetParametersSchemaReturnsNormalizedSchema) {
    ASSERT_TRUE(registry.register_tool("add", "Add", {"a", "b"}, add).has_value());

    auto schema = registry.get_parameters_schema("add");
    ASSERT_TRUE(schema.has_value());
    EXPECT_EQ((*schema)["required"], json::array({"a", "b"}));
    EXPECT_EQ((*schema)["additionalProperties"], false);

    auto missing = registry.get_parameters_schema("nonexistent");
    EXPECT_FALSE(missing.has_value());
}

TEST_F(ToolRegistryTest, LambdaRegistration) {
    auto result =
        registry.register_tool("double_it", "Double a number", {"x"}, [](int x) { return x * 2; });
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    auto invoke_result = registry.invoke("double_it", {{"x", 5}});
    ASSERT_TRUE(invoke_result.has_value());
    EXPECT_EQ((*invoke_result)["result"], 10);
}

TEST_F(ToolRegistryTest, OverwriteExistingPreservesOrder) {
    ASSERT_TRUE(registry.register_tool("add", "Add v1", {"a", "b"}, add).has_value());
    ASSERT_TRUE(registry.register_tool("greet", "Greet", {"name"}, greet).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add v2", {"a", "b"}, add).has_value());

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"add", "greet"}));

    auto schema = registry.get_tool_schema("add");
    EXPECT_EQ(schema["function"]["description"], "Add v2");
}

TEST_F(ToolRegistryTest, RegisterToolsBatchAddsAllTools) {
    auto def1 = zoo::tools::detail::make_tool_definition("add", "Add",
                                                         std::vector<std::string>{"a", "b"}, add);
    auto def2 = zoo::tools::detail::make_tool_definition("greet", "Greet",
                                                         std::vector<std::string>{"name"}, greet);
    ASSERT_TRUE(def1.has_value());
    ASSERT_TRUE(def2.has_value());

    std::vector<zoo::tools::ToolDefinition> definitions;
    definitions.push_back(std::move(*def1));
    definitions.push_back(std::move(*def2));

    auto result = registry.register_tools(std::move(definitions));
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(registry.size(), 2u);
    EXPECT_TRUE(registry.has_tool("add"));
    EXPECT_TRUE(registry.has_tool("greet"));

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"add", "greet"}));
}

TEST_F(ToolRegistryTest, RegisterToolsBatchEmptyIsNoOp) {
    auto result = registry.register_tools({});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(registry.size(), 0u);
}

TEST_F(ToolRegistryTest, InvokeDoesNotBlockConcurrentReads) {
    auto entered = std::make_shared<std::promise<void>>();
    auto entered_future = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto release_future = release->get_future().share();

    zoo::tools::ToolHandler slow_handler = [entered,
                                            release_future](const json&) -> zoo::Expected<json> {
        entered->set_value();
        release_future.wait();
        return json{{"result", "done"}};
    };

    ASSERT_TRUE(registry
                    .register_tool("slow", "A slow tool",
                                   json{{"type", "object"}, {"properties", json::object()}},
                                   std::move(slow_handler))
                    .has_value());

    std::thread invoker([this] { registry.invoke("slow", json::object()); });

    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);

    auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(registry.has_tool("slow"));
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(elapsed, std::chrono::milliseconds(100));

    release->set_value();
    invoker.join();
}
