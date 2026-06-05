/**
 * @file test_tool_registry.cpp
 * @brief Unit tests for tool schema registration and normalization.
 */

#include "zoo/tools/registry.hpp"
#include "zoo/tools/validation.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

using json = nlohmann::json;
namespace detail = zoo::tools::detail;

namespace {

json add_schema() {
    return {{"type", "object"},
            {"properties", {{"a", {{"type", "integer"}}}, {"b", {{"type", "integer"}}}}},
            {"required", json::array({"a", "b"})},
            {"additionalProperties", false}};
}

json greet_schema() {
    return {{"type", "object"},
            {"properties", {{"name", {{"type", "string"}}}}},
            {"required", json::array({"name"})},
            {"additionalProperties", false}};
}

} // namespace

/// Shared fixture that provides a fresh tool registry for each test.
class ToolRegistryTest : public ::testing::Test {
  protected:
    zoo::tools::ToolRegistry registry;
};

TEST_F(ToolRegistryTest, RegisterSchemaNormalizesModelFacingSpec) {
    ASSERT_TRUE(registry.register_tool("add", "Add two integers", add_schema()).has_value());

    auto schema = registry.get_tool_schema("add");
    ASSERT_TRUE(schema.is_object());
    EXPECT_EQ(schema["function"]["name"], "add");
    EXPECT_EQ(schema["function"]["description"], "Add two integers");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["a"]["type"], "integer");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["b"]["type"], "integer");
    EXPECT_EQ(schema["function"]["parameters"]["required"], json::array({"a", "b"}));
    EXPECT_EQ(schema["function"]["parameters"]["additionalProperties"], false);

    auto spec = registry.get_tool_spec("add");
    ASSERT_TRUE(spec.has_value());
    EXPECT_EQ(spec->name, "add");
    EXPECT_EQ(spec->description, "Add two integers");
    EXPECT_EQ(spec->parameters_schema, schema["function"]["parameters"]);
}

TEST_F(ToolRegistryTest, RegisterToolSpecCanonicalizesSchema) {
    zoo::ToolSpec spec;
    spec.name = "greet";
    spec.description = "Greet someone";
    spec.parameters_schema = {{"type", "object"},
                              {"properties", {{"name", {{"type", "string"}}}}},
                              {"required", json::array({"name"})}};

    ASSERT_TRUE(registry.register_tool(std::move(spec)).has_value());

    auto stored = registry.get_tool_spec("greet");
    ASSERT_TRUE(stored.has_value());
    EXPECT_EQ(stored->parameters_schema["additionalProperties"], false);
    EXPECT_EQ(stored->parameters_schema["required"], json::array({"name"}));
}

TEST_F(ToolRegistryTest, ManualSchemaRegistrationNormalizesOptionalAndEnumFields) {
    json schema = {{"type", "object"},
                   {"properties",
                    {{"query", {{"type", "string"}, {"description", "Search query"}}},
                     {"limit", {{"type", "integer"}, {"enum", json::array({5, 10, 20})}}}}},
                   {"required", json::array({"query"})},
                   {"additionalProperties", false}};

    auto result = registry.register_tool("search", "Search documents", schema);
    ASSERT_TRUE(result.has_value()) << result.error().to_string();

    auto parameters = registry.get_tool_parameters("search");
    ASSERT_TRUE(parameters.has_value());
    ASSERT_EQ(parameters->size(), 2u);

    EXPECT_EQ((*parameters)[0].name, "query");
    EXPECT_TRUE((*parameters)[0].required);
    EXPECT_EQ((*parameters)[1].name, "limit");
    EXPECT_FALSE((*parameters)[1].required);
    EXPECT_EQ((*parameters)[1].enum_values, json::array({5, 10, 20}).get<std::vector<json>>());
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

    auto result = registry.register_tool("search_documents", "Search with nested query", schema);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsUnsupportedKeywords) {
    json schema = {{"type", "object"},
                   {"properties", {{"limit", {{"type", "integer"}, {"minimum", 1}}}}}};

    auto result = registry.register_tool("bounded_limit", "Schema with unsupported bounds", schema);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("minimum"), std::string::npos);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsUnsupportedRootKeywords) {
    json schema = {{"type", "object"}, {"properties", json::object()}, {"oneOf", json::array()}};

    auto result =
        registry.register_tool("root_keyword", "Schema with unsupported root keyword", schema);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("oneOf"), std::string::npos);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsRefKeyword) {
    json schema = {{"type", "object"},
                   {"properties", {{"data", {{"$ref", "#/definitions/Data"}}}}},
                   {"required", json::array({"data"})}};

    auto result = registry.register_tool("ref_tool", "Schema with $ref", schema);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST_F(ToolRegistryTest, ManualSchemaRejectsArrayType) {
    json schema = {
        {"type", "object"},
        {"properties", {{"items", {{"type", "array"}, {"items", {{"type", "string"}}}}}}},
        {"required", json::array({"items"})}};

    auto result = registry.register_tool("array_tool", "Schema with array", schema);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST_F(ToolRegistryTest, GetAllSchemasUsesRegistrationOrder) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet", greet_schema()).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add", add_schema()).has_value());

    auto schemas = registry.get_all_schemas();
    ASSERT_EQ(schemas.size(), 2u);
    EXPECT_EQ(schemas[0]["function"]["name"], "greet");
    EXPECT_EQ(schemas[1]["function"]["name"], "add");
}

TEST_F(ToolRegistryTest, GetToolNamesUsesRegistrationOrder) {
    ASSERT_TRUE(registry.register_tool("greet", "Greet", greet_schema()).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add", add_schema()).has_value());

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"greet", "add"}));
}

TEST_F(ToolRegistryTest, GetParametersSchemaReturnsNormalizedSchema) {
    ASSERT_TRUE(registry.register_tool("add", "Add", add_schema()).has_value());

    auto schema = registry.get_parameters_schema("add");
    ASSERT_TRUE(schema.has_value());
    EXPECT_EQ((*schema)["required"], json::array({"a", "b"}));
    EXPECT_EQ((*schema)["additionalProperties"], false);

    auto missing = registry.get_parameters_schema("nonexistent");
    EXPECT_FALSE(missing.has_value());
}

TEST_F(ToolRegistryTest, OverwriteExistingPreservesOrder) {
    ASSERT_TRUE(registry.register_tool("add", "Add v1", add_schema()).has_value());
    ASSERT_TRUE(registry.register_tool("greet", "Greet", greet_schema()).has_value());
    ASSERT_TRUE(registry.register_tool("add", "Add v2", add_schema()).has_value());

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"add", "greet"}));

    auto schema = registry.get_tool_schema("add");
    EXPECT_EQ(schema["function"]["description"], "Add v2");
}

TEST_F(ToolRegistryTest, RegisterToolsBatchAddsAllTools) {
    std::vector<zoo::ToolSpec> specs;
    specs.push_back(zoo::ToolSpec{"add", "Add", add_schema()});
    specs.push_back(zoo::ToolSpec{"greet", "Greet", greet_schema()});

    auto result = registry.register_tools(std::move(specs));
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(registry.size(), 2u);
    EXPECT_TRUE(registry.has_tool("add"));
    EXPECT_TRUE(registry.has_tool("greet"));

    auto names = registry.get_tool_names();
    EXPECT_EQ(names, std::vector<std::string>({"add", "greet"}));
}

TEST_F(ToolRegistryTest, RegisterToolsBatchValidatesBeforeMutatingRegistry) {
    ASSERT_TRUE(registry.register_tool("add", "Add", add_schema()).has_value());

    std::vector<zoo::ToolSpec> specs;
    specs.push_back(zoo::ToolSpec{"greet", "Greet", greet_schema()});
    specs.push_back(zoo::ToolSpec{"broken", "Broken", json{{"type", "array"}}});

    auto result = registry.register_tools(std::move(specs));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);

    EXPECT_EQ(registry.size(), 1u);
    EXPECT_TRUE(registry.has_tool("add"));
    EXPECT_FALSE(registry.has_tool("greet"));
    EXPECT_FALSE(registry.has_tool("broken"));
}

TEST_F(ToolRegistryTest, RegisterToolsBatchEmptyIsNoOp) {
    auto result = registry.register_tools({});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(registry.size(), 0u);
}

// ---------------------------------------------------------------------------
// Tests for the detail free functions moved out of the public header.
// All of these paths are exercised indirectly through ToolRegistry::register_tool
// in the ToolRegistryTest suite above, but the branches below were only reachable
// via specific schema combinations that the high-level tests did not cover.
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

TEST(JsonMatchesTypeTest, IntegralFloatsMatchInteger) {
    EXPECT_TRUE(detail::json_matches_type(json(3.0), zoo::tools::ToolValueType::Integer));
    EXPECT_TRUE(detail::json_matches_type(json(-42.0), zoo::tools::ToolValueType::Integer));
    EXPECT_TRUE(detail::json_matches_type(json(0.0), zoo::tools::ToolValueType::Integer));
}

TEST(JsonMatchesTypeTest, FractionalFloatsDoNotMatchInteger) {
    EXPECT_FALSE(detail::json_matches_type(json(3.14), zoo::tools::ToolValueType::Integer));
    EXPECT_FALSE(detail::json_matches_type(json(-1.5), zoo::tools::ToolValueType::Integer));
}

TEST(JsonMatchesTypeTest, NonFiniteFloatsDoNotMatchInteger) {
    EXPECT_FALSE(detail::json_matches_type(json(std::numeric_limits<double>::infinity()),
                                           zoo::tools::ToolValueType::Integer));
    EXPECT_FALSE(detail::json_matches_type(json(std::numeric_limits<double>::quiet_NaN()),
                                           zoo::tools::ToolValueType::Integer));
}

TEST(JsonMatchesTypeTest, OutOfRangeNumbersDoNotMatchInteger) {
    const auto above_int_max = static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
    const auto below_int_min = static_cast<std::int64_t>(std::numeric_limits<int>::min()) - 1;

    EXPECT_FALSE(
        detail::json_matches_type(json(above_int_max), zoo::tools::ToolValueType::Integer));
    EXPECT_FALSE(
        detail::json_matches_type(json(below_int_min), zoo::tools::ToolValueType::Integer));
    EXPECT_FALSE(detail::json_matches_type(json(std::numeric_limits<std::uint64_t>::max()),
                                           zoo::tools::ToolValueType::Integer));
    EXPECT_FALSE(
        detail::json_matches_type(json(static_cast<double>(std::numeric_limits<int>::max()) + 1.0),
                                  zoo::tools::ToolValueType::Integer));
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

// --- normalize_tool_parameters: schema-level validation --------------------

TEST(NormalizeToolParametersTest, RejectsNonObjectSchema) {
    auto result = detail::normalize_tool_parameters("t", json::array());
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsMissingTopLevelType) {
    json schema = {{"properties", json::object()}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsNonStringTopLevelType) {
    json schema = {{"type", 42}, {"properties", json::object()}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsNonObjectTopLevelType) {
    json schema = {{"type", "array"}, {"properties", json::object()}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsMissingProperties) {
    json schema = {{"type", "object"}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsNonObjectProperties) {
    json schema = {{"type", "object"}, {"properties", json::array()}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsAdditionalPropertiesTrue) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"additionalProperties", true}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsAdditionalPropertiesNonBoolean) {
    // "additionalProperties": {} is a common JSON Schema pattern but is unsupported here
    json schema = {{"type", "object"},
                   {"properties", json::object()},
                   {"additionalProperties", json::object()}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, AcceptsAdditionalPropertiesFalse) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"additionalProperties", false}};
    auto result = detail::normalize_tool_parameters("t", schema);
    EXPECT_TRUE(result.has_value());
}

TEST(NormalizeToolParametersTest, RejectsNonArrayRequired) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"required", json::object()}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsNonStringRequiredEntry) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "integer"}}}}},
                   {"required", json::array({42})}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsDuplicateRequiredEntry) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "integer"}}}}},
                   {"required", json::array({"x", "x"})}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("duplicate"), std::string::npos);
}

TEST(NormalizeToolParametersTest, RejectsRequiredEntryNotInProperties) {
    json schema = {
        {"type", "object"}, {"properties", json::object()}, {"required", json::array({"ghost"})}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
    EXPECT_NE(result.error().message.find("ghost"), std::string::npos);
}

TEST(NormalizeToolParametersTest, RejectsNonObjectPropertyValue) {
    json schema = {{"type", "object"}, {"properties", {{"x", "not-an-object"}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsPropertyWithMissingType) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"description", "no type"}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsPropertyWithNonStringType) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"type", 99}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsPropertyWithNonStringDescription) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "string"}, {"description", 123}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsPropertyWithNonArrayEnum) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "string"}, {"enum", "not-array"}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, RejectsEnumValueTypeMismatch) {
    // "x" is declared integer but the enum contains a string
    json schema = {
        {"type", "object"},
        {"properties", {{"x", {{"type", "integer"}, {"enum", json::array({1, "two"})}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}

TEST(NormalizeToolParametersTest, ParameterOrderIsRequiredThenOptional) {
    // "b" is required, "a" comes first alphabetically; required should still win.
    json schema = {{"type", "object"},
                   {"properties", {{"a", {{"type", "string"}}}, {"b", {{"type", "integer"}}}}},
                   {"required", json::array({"b"})}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    EXPECT_EQ((*result)[0].name, "b");
    EXPECT_TRUE((*result)[0].required);
    EXPECT_EQ((*result)[1].name, "a");
    EXPECT_FALSE((*result)[1].required);
}

TEST(NormalizeToolParametersTest, SetsDescriptionOnParameter) {
    json schema = {{"type", "object"},
                   {"properties", {{"x", {{"type", "string"}, {"description", "the x value"}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].description, "the x value");
}

TEST(NormalizeToolParametersTest, AcceptsNumberAndBooleanTypes) {
    json schema = {
        {"type", "object"},
        {"properties", {{"ratio", {{"type", "number"}}}, {"flag", {{"type", "boolean"}}}}}};
    auto result = detail::normalize_tool_parameters("t", schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 2u);
    // nlohmann preserves insertion order which is alphabetical for object literals
    EXPECT_EQ((*result)[0].type, zoo::tools::ToolValueType::Boolean);
    EXPECT_EQ((*result)[1].type, zoo::tools::ToolValueType::Number);
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

TEST(NormalizeSchemaTest, DelegatesToNormalizeToolParameters) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"type", "integer"}}}}}};
    auto result = detail::normalize_schema(schema);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ((*result)[0].name, "x");
}

TEST(NormalizeSchemaTest, PropagatesErrorFromNormalizeToolParameters) {
    auto result = detail::normalize_schema(json("not-an-object"));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, zoo::ErrorCode::InvalidToolSchema);
}
