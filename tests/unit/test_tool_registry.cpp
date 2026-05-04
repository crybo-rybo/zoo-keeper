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
namespace detail = zoo::tools::detail;

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

TEST_F(ToolRegistryTest, ManualSchemaRejectsUnsupportedRootKeywords) {
    json schema = {{"type", "object"}, {"properties", json::object()}, {"oneOf", json::array()}};

    auto result =
        registry.register_tool("root_keyword", "Schema with unsupported root keyword", schema,
                               [](const json&) -> zoo::Expected<json> { return json::object(); });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("oneOf"), std::string::npos);
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
        {"properties", {{"items", {{"type", "array"}}}}},
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

TEST(ToolDefinitionFactoryTest, TypedCallableBuildsMetadataAndHandler) {
    auto definition = zoo::tools::make_tool_definition("add", "Add two integers", {"a", "b"},
                                                       [](int a, int b) { return a + b; });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();

    EXPECT_EQ(definition->metadata.name, "add");
    ASSERT_EQ(definition->metadata.parameters.size(), 2u);
    EXPECT_EQ(definition->metadata.parameters[0].type, zoo::tools::ToolValueType::Integer);
    EXPECT_EQ(definition->metadata.parameters_schema["required"], json::array({"a", "b"}));

    auto result = definition->handler({{"a", 2}, {"b", 5}});
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ((*result)["result"], 7);
}

TEST(ToolDefinitionFactoryTest, JsonSchemaBuildsMetadataAndHandler) {
    json schema = {{"type", "object"},
                   {"properties",
                    {{"query", {{"type", "string"}}},
                     {"limit", {{"type", "integer"}, {"enum", json::array({5, 10})}}}}},
                   {"required", json::array({"query"})},
                   {"additionalProperties", false}};

    auto definition = zoo::tools::make_tool_definition(
        "search", "Search documents", schema, [](const json& args) -> zoo::Expected<json> {
            return json{{"query", args.at("query")}, {"limit", args.value("limit", 5)}};
        });
    ASSERT_TRUE(definition.has_value()) << definition.error().to_string();

    ASSERT_EQ(definition->metadata.parameters.size(), 2u);
    EXPECT_EQ(definition->metadata.parameters[0].name, "query");
    EXPECT_TRUE(definition->metadata.parameters[0].required);
    EXPECT_EQ(definition->metadata.parameters[1].name, "limit");
    EXPECT_FALSE(definition->metadata.parameters[1].required);

    auto result = definition->handler({{"query", "llama"}});
    ASSERT_TRUE(result.has_value()) << result.error().to_string();
    EXPECT_EQ((*result)["limit"], 5);
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
    auto def1 =
        zoo::tools::make_tool_definition("add", "Add", std::vector<std::string>{"a", "b"}, add);
    auto def2 =
        zoo::tools::make_tool_definition("greet", "Greet", std::vector<std::string>{"name"}, greet);
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

// ---------------------------------------------------------------------------
// Tests for the detail free functions moved out of the public header.
// All of these paths are exercised indirectly through ToolRegistry::register_tool
// in the ToolRegistryTest suite above, but the branches below were only reachable
// via specific combinations that the high-level tests did not cover.
// ---------------------------------------------------------------------------

// --- parse_tool_value_type -------------------------------------------------

TEST(ParseToolValueTypeTest, AcceptsAllFourPrimitives) {
    EXPECT_EQ(*detail::parse_tool_value_type("integer"), zoo::tools::ToolValueType::Integer);
    EXPECT_EQ(*detail::parse_tool_value_type("number"), zoo::tools::ToolValueType::Number);
    EXPECT_EQ(*detail::parse_tool_value_type("string"), zoo::tools::ToolValueType::String);
    EXPECT_EQ(*detail::parse_tool_value_type("boolean"), zoo::tools::ToolValueType::Boolean);
}

TEST(ParseToolValueTypeTest, RejectsUnknownTypeString) {
    auto result = detail::parse_tool_value_type("widget");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("widget"), std::string::npos);
}

TEST(ParseToolValueTypeTest, RejectsEmptyString) {
    auto result = detail::parse_tool_value_type("");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

// --- json_matches_type -----------------------------------------------------

TEST(JsonMatchesTypeTest, IntegerMatchesInteger) {
    EXPECT_TRUE(detail::json_matches_type(json(42), zoo::tools::ToolValueType::Integer));
}

TEST(JsonMatchesTypeTest, FloatDoesNotMatchInteger) {
    // JSON floats are not integers
    EXPECT_FALSE(detail::json_matches_type(json(3.14), zoo::tools::ToolValueType::Integer));
}

TEST(JsonMatchesTypeTest, IntegerMatchesNumber) {
    // integers are numbers in JSON
    EXPECT_TRUE(detail::json_matches_type(json(1), zoo::tools::ToolValueType::Number));
}

TEST(JsonMatchesTypeTest, FloatMatchesNumber) {
    EXPECT_TRUE(detail::json_matches_type(json(1.5), zoo::tools::ToolValueType::Number));
}

TEST(JsonMatchesTypeTest, StringMatchesString) {
    EXPECT_TRUE(detail::json_matches_type(json("hello"), zoo::tools::ToolValueType::String));
}

TEST(JsonMatchesTypeTest, NonStringDoesNotMatchString) {
    EXPECT_FALSE(detail::json_matches_type(json(0), zoo::tools::ToolValueType::String));
}

TEST(JsonMatchesTypeTest, BoolMatchesBoolean) {
    EXPECT_TRUE(detail::json_matches_type(json(true), zoo::tools::ToolValueType::Boolean));
    EXPECT_TRUE(detail::json_matches_type(json(false), zoo::tools::ToolValueType::Boolean));
}

TEST(JsonMatchesTypeTest, NonBoolDoesNotMatchBoolean) {
    EXPECT_FALSE(detail::json_matches_type(json(1), zoo::tools::ToolValueType::Boolean));
}

// --- validate_enum_values --------------------------------------------------

TEST(ValidateEnumValuesTest, AcceptsMatchingStringEnumValues) {
    std::vector<json> values = {json("a"), json("b"), json("c")};
    auto result = detail::validate_enum_values(values, zoo::tools::ToolValueType::String, "t", "p");
    EXPECT_TRUE(result.has_value());
}

TEST(ValidateEnumValuesTest, AcceptsMatchingBooleanEnumValues) {
    std::vector<json> values = {json(true), json(false)};
    auto result =
        detail::validate_enum_values(values, zoo::tools::ToolValueType::Boolean, "t", "p");
    EXPECT_TRUE(result.has_value());
}

TEST(ValidateEnumValuesTest, AcceptsMatchingNumberEnumValues) {
    std::vector<json> values = {json(1.0), json(2.5)};
    auto result = detail::validate_enum_values(values, zoo::tools::ToolValueType::Number, "t", "p");
    EXPECT_TRUE(result.has_value());
}

TEST(ValidateEnumValuesTest, RejectsMismatchedEnumValueType) {
    // Enum declares integer type but supplies a string value
    std::vector<json> values = {json(1), json("two")};
    auto result =
        detail::validate_enum_values(values, zoo::tools::ToolValueType::Integer, "tool", "param");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("param"), std::string::npos);
    EXPECT_NE(result.error().message.find("tool"), std::string::npos);
}

TEST(ValidateEnumValuesTest, AcceptsEmptyEnumList) {
    std::vector<json> values;
    auto result =
        detail::validate_enum_values(values, zoo::tools::ToolValueType::Integer, "t", "p");
    EXPECT_TRUE(result.has_value());
}

// --- normalize_manual_tool_metadata: schema-level validation ---------------

TEST(NormalizeManualToolMetadataTest, RejectsNonObjectSchema) {
    auto result = detail::normalize_manual_tool_metadata("t", "d", json::array());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsMissingTopLevelType) {
    json schema = {{"properties", json::object()}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsNonStringTopLevelType) {
    json schema = {{"type", 42}, {"properties", json::object()}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsNonObjectTopLevelType) {
    json schema = {{"type", "array"}, {"properties", json::object()}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsMissingProperties) {
    json schema = {{"type", "object"}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsNonObjectProperties) {
    json schema = {{"type", "object"}, {"properties", json::array()}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsAdditionalPropertiesTrue) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"additionalProperties", true}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsAdditionalPropertiesNonBoolean) {
    // "additionalProperties": {} is a common JSON Schema pattern but is unsupported here
    json schema = {{"type", "object"},
                   {"properties", json::object()},
                   {"additionalProperties", json::object()}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, AcceptsAdditionalPropertiesFalse) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"additionalProperties", false}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    EXPECT_TRUE(result.has_value());
}

TEST(NormalizeManualToolMetadataTest, RejectsNonArrayRequired) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"required", json::object()}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsNonStringRequiredEntry) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "integer"}}}}},
                   {"required", json::array({42})}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsDuplicateRequiredEntry) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "integer"}}}}},
                   {"required", json::array({"x", "x"})}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("duplicate"), std::string::npos);
}

TEST(NormalizeManualToolMetadataTest, RejectsRequiredEntryNotInProperties) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"required", json::array({"ghost"})}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("ghost"), std::string::npos);
}

TEST(NormalizeManualToolMetadataTest, RejectsNonObjectPropertyValue) {
    json schema = {{"type", "object"}, {"properties", {{"x", "not-an-object"}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsPropertyWithMissingType) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"description", "no type"}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsPropertyWithNonStringType) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"type", 99}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsPropertyWithNonStringDescription) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "string"}, {"description", 123}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsPropertyWithNonArrayEnum) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "string"}, {"enum", "not-array"}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, RejectsEnumValueTypeMismatch) {
    // "x" is declared integer but the enum contains a string
    json schema = {
        {"type", "object"},
        {"properties", {{"x", {{"type", "integer"}, {"enum", json::array({1, "two"})}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeManualToolMetadataTest, ParameterOrderIsRequiredThenOptional) {
    // "b" is required, "a" comes first alphabetically — required should still win
    json schema = {{"type", "object"},
                   {"properties", {{"a", {{"type", "string"}}}, {"b", {{"type", "integer"}}}}},
                   {"required", json::array({"b"})}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->parameters.size(), 2u);
    EXPECT_EQ(result->parameters[0].name, "b");
    EXPECT_TRUE(result->parameters[0].required);
    EXPECT_EQ(result->parameters[1].name, "a");
    EXPECT_FALSE(result->parameters[1].required);
}

TEST(NormalizeManualToolMetadataTest, SetsDescriptionOnParameter) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "string"}, {"description", "the x value"}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->parameters.size(), 1u);
    EXPECT_EQ(result->parameters[0].description, "the x value");
}

TEST(NormalizeManualToolMetadataTest, AcceptsNumberAndBooleanTypes) {
    json schema = {
        {"type", "object"},
        {"properties", {{"ratio", {{"type", "number"}}}, {"flag", {{"type", "boolean"}}}}}};
    auto result = detail::normalize_manual_tool_metadata("t", "d", schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->parameters.size(), 2u);
    // nlohmann preserves insertion order which is alphabetical for object literals
    EXPECT_EQ(result->parameters[0].type, zoo::tools::ToolValueType::Boolean);
    EXPECT_EQ(result->parameters[1].type, zoo::tools::ToolValueType::Number);
}

// --- build_parameters_schema -----------------------------------------------

TEST(BuildParametersSchemaTest, EmptyParameterListProducesEmptySchema) {
    auto schema = detail::build_parameters_schema({});
    EXPECT_EQ(schema["type"], "object");
    EXPECT_TRUE(schema["properties"].empty());
    EXPECT_TRUE(schema["required"].empty());
    EXPECT_EQ(schema["additionalProperties"], false);
}

TEST(BuildParametersSchemaTest, DescriptionAppearsInPropertySchema) {
    zoo::tools::ToolParameter p;
    p.name = "q";
    p.type = zoo::tools::ToolValueType::String;
    p.required = true;
    p.description = "the query";

    auto schema = detail::build_parameters_schema({p});
    EXPECT_EQ(schema["properties"]["q"]["description"], "the query");
    EXPECT_EQ(schema["properties"]["q"]["type"], "string");
    EXPECT_EQ(schema["required"], json::array({"q"}));
}

TEST(BuildParametersSchemaTest, NoDescriptionKeyWhenDescriptionIsEmpty) {
    zoo::tools::ToolParameter p;
    p.name = "n";
    p.type = zoo::tools::ToolValueType::Integer;
    p.required = false;

    auto schema = detail::build_parameters_schema({p});
    EXPECT_FALSE(schema["properties"]["n"].contains("description"));
}

TEST(BuildParametersSchemaTest, EnumAppearsInPropertySchema) {
    zoo::tools::ToolParameter p;
    p.name = "size";
    p.type = zoo::tools::ToolValueType::String;
    p.required = false;
    p.enum_values = {json("sm"), json("md"), json("lg")};

    auto schema = detail::build_parameters_schema({p});
    EXPECT_EQ(schema["properties"]["size"]["enum"], json::array({"sm", "md", "lg"}));
}

TEST(BuildParametersSchemaTest, OptionalParamNotAddedToRequired) {
    zoo::tools::ToolParameter p;
    p.name = "opt";
    p.type = zoo::tools::ToolValueType::Boolean;
    p.required = false;

    auto schema = detail::build_parameters_schema({p});
    EXPECT_TRUE(schema["required"].empty());
}

// --- normalize_schema (thin wrapper) ---------------------------------------

TEST(NormalizeSchemaTest, DelegatesToNormalizeManualToolMetadata) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"type", "integer"}}}}}};
    auto result = detail::normalize_schema(schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].name, "x");
}

TEST(NormalizeSchemaTest, PropagatesErrorFromNormalizeManualToolMetadata) {
    auto result = detail::normalize_schema(json("not-an-object"));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}
