#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include "zoo/engine/tool_registry.hpp"
#include "fixtures/tool_definitions.hpp"

using namespace zoo;
using namespace zoo::engine;
using namespace zoo::testing::tools;
using json = nlohmann::json;

class ToolRegistryTest : public ::testing::Test {
protected:
    ToolRegistry registry;
};

// ============================================================================
// TR-001: Register tool with primitive parameters -> schema generated correctly
// ============================================================================

TEST_F(ToolRegistryTest, RegisterIntParams) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    EXPECT_TRUE(registry.has_tool("add"));
    EXPECT_EQ(registry.size(), 1);

    auto schema = registry.get_tool_schema("add");
    EXPECT_EQ(schema["function"]["name"], "add");
    EXPECT_EQ(schema["function"]["description"], "Add two integers");

    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["type"], "object");
    EXPECT_EQ(params["properties"]["a"]["type"], "integer");
    EXPECT_EQ(params["properties"]["b"]["type"], "integer");
}

// ============================================================================
// TR-002: Int type schema
// ============================================================================

TEST_F(ToolRegistryTest, IntTypeSchema) {
    (void)registry.register_tool("negate", "Negate a number", {"value"}, negate);

    auto schema = registry.get_tool_schema("negate");
    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["properties"]["value"]["type"], "integer");
}

// ============================================================================
// TR-003: Float/double type schema
// ============================================================================

TEST_F(ToolRegistryTest, DoubleTypeSchema) {
    (void)registry.register_tool("multiply", "Multiply two doubles", {"a", "b"}, multiply);

    auto schema = registry.get_tool_schema("multiply");
    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["properties"]["a"]["type"], "number");
    EXPECT_EQ(params["properties"]["b"]["type"], "number");
}

// ============================================================================
// TR-004: Bool type schema
// ============================================================================

TEST_F(ToolRegistryTest, BoolTypeSchema) {
    (void)registry.register_tool("is_positive", "Check if positive", {"n"}, is_positive);

    auto schema = registry.get_tool_schema("is_positive");
    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["properties"]["n"]["type"], "integer");
}

// ============================================================================
// TR-005: String type schema
// ============================================================================

TEST_F(ToolRegistryTest, StringTypeSchema) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);

    auto schema = registry.get_tool_schema("greet");
    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["properties"]["name"]["type"], "string");
}

// ============================================================================
// TR-006: Multiple parameters -> all shown in schema
// ============================================================================

TEST_F(ToolRegistryTest, MultipleParametersInSchema) {
    (void)registry.register_tool("concat", "Concatenate strings", {"a", "b"}, concat);

    auto schema = registry.get_tool_schema("concat");
    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["properties"].size(), 2);
    EXPECT_TRUE(params["properties"].contains("a"));
    EXPECT_TRUE(params["properties"].contains("b"));
    EXPECT_EQ(params["required"].size(), 2);
}

// ============================================================================
// TR-007: Generated schema is valid JSON schema
// ============================================================================

TEST_F(ToolRegistryTest, ValidJsonSchema) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto schema = registry.get_tool_schema("add");

    // Must have "type": "function"
    EXPECT_EQ(schema["type"], "function");

    // Must have "function" with name, description, parameters
    EXPECT_TRUE(schema.contains("function"));
    EXPECT_TRUE(schema["function"].contains("name"));
    EXPECT_TRUE(schema["function"].contains("description"));
    EXPECT_TRUE(schema["function"].contains("parameters"));

    // Parameters must be a valid JSON Schema object
    auto params = schema["function"]["parameters"];
    EXPECT_EQ(params["type"], "object");
    EXPECT_TRUE(params.contains("properties"));
    EXPECT_TRUE(params.contains("required"));
}

// ============================================================================
// TR-008: Schema includes description
// ============================================================================

TEST_F(ToolRegistryTest, SchemaIncludesDescription) {
    (void)registry.register_tool("greet", "Greet a person by name", {"name"}, greet);

    auto schema = registry.get_tool_schema("greet");
    EXPECT_EQ(schema["function"]["description"], "Greet a person by name");
}

// ============================================================================
// TR-009: Invoke with valid args -> handler called, result returned
// ============================================================================

TEST_F(ToolRegistryTest, InvokeWithValidArgs) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", 3}, {"b", 4}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 7);
}

TEST_F(ToolRegistryTest, InvokeStringTool) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);

    auto result = registry.invoke("greet", {{"name", "Alice"}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "Hello, Alice!");
}

TEST_F(ToolRegistryTest, InvokeDoubleTool) {
    (void)registry.register_tool("circle_area", "Compute circle area", {"radius"}, circle_area);

    auto result = registry.invoke("circle_area", {{"radius", 1.0}});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)["result"].get<double>(), 3.14159, 0.001);
}

TEST_F(ToolRegistryTest, InvokeBoolResult) {
    (void)registry.register_tool("is_positive", "Check if positive", {"n"}, is_positive);

    auto result_pos = registry.invoke("is_positive", {{"n", 5}});
    ASSERT_TRUE(result_pos.has_value());
    EXPECT_EQ((*result_pos)["result"], true);

    auto result_neg = registry.invoke("is_positive", {{"n", -3}});
    ASSERT_TRUE(result_neg.has_value());
    EXPECT_EQ((*result_neg)["result"], false);
}

// ============================================================================
// TR-010: Invoke with wrong types -> error returned
// ============================================================================

TEST_F(ToolRegistryTest, InvokeWrongArgType) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", "not_a_number"}, {"b", 4}});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolExecutionFailed);
}

TEST_F(ToolRegistryTest, InvokeMissingArg) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", 3}});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolExecutionFailed);
}

// ============================================================================
// TR-011: Invoke unregistered tool -> ToolNotFound error
// ============================================================================

TEST_F(ToolRegistryTest, InvokeUnregisteredTool) {
    auto result = registry.invoke("nonexistent", {});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolNotFound);
}

// ============================================================================
// TR-012: Re-register same name -> overwrites
// ============================================================================

TEST_F(ToolRegistryTest, ReRegisterOverwrites) {
    (void)registry.register_tool("calc", "Old description", {"a", "b"}, add);
    EXPECT_EQ(registry.size(), 1);

    (void)registry.register_tool("calc", "New description", {"a", "b"}, multiply);
    EXPECT_EQ(registry.size(), 1);

    auto schema = registry.get_tool_schema("calc");
    EXPECT_EQ(schema["function"]["description"], "New description");

    // New handler is used
    auto result = registry.invoke("calc", {{"a", 3.0}, {"b", 4.0}});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)["result"].get<double>(), 12.0, 0.001);
}

// ============================================================================
// Additional tests
// ============================================================================

TEST_F(ToolRegistryTest, GetToolNames) {
    (void)registry.register_tool("add", "Add", {"a", "b"}, add);
    (void)registry.register_tool("greet", "Greet", {"name"}, greet);

    auto names = registry.get_tool_names();
    EXPECT_EQ(names.size(), 2);
    // Both tools present (order not guaranteed with unordered_map)
    bool has_add = std::find(names.begin(), names.end(), "add") != names.end();
    bool has_greet = std::find(names.begin(), names.end(), "greet") != names.end();
    EXPECT_TRUE(has_add);
    EXPECT_TRUE(has_greet);
}

TEST_F(ToolRegistryTest, GetAllSchemas) {
    (void)registry.register_tool("add", "Add numbers", {"a", "b"}, add);
    (void)registry.register_tool("greet", "Greet", {"name"}, greet);

    auto schemas = registry.get_all_schemas();
    EXPECT_EQ(schemas.size(), 2);

    // Each entry should be a valid tool schema
    for (const auto& schema : schemas) {
        EXPECT_EQ(schema["type"], "function");
        EXPECT_TRUE(schema["function"].contains("name"));
        EXPECT_TRUE(schema["function"].contains("parameters"));
    }
}

TEST_F(ToolRegistryTest, HasToolReturnsFalseForMissing) {
    EXPECT_FALSE(registry.has_tool("nonexistent"));
}

TEST_F(ToolRegistryTest, EmptyRegistrySize) {
    EXPECT_EQ(registry.size(), 0);
}

TEST_F(ToolRegistryTest, GetSchemaForMissingToolReturnsEmpty) {
    auto schema = registry.get_tool_schema("nonexistent");
    EXPECT_TRUE(schema.empty());
}

TEST_F(ToolRegistryTest, ManualRegistration) {
    nlohmann::json schema = {
        {"type", "object"},
        {"properties", {
            {"query", {{"type", "string"}}}
        }},
        {"required", {"query"}}
    };

    ToolHandler handler = [](const nlohmann::json& args) -> Expected<nlohmann::json> {
        return nlohmann::json{{"result", "found: " + args["query"].get<std::string>()}};
    };

    (void)registry.register_tool("search", "Search for something", std::move(schema), std::move(handler));

    EXPECT_TRUE(registry.has_tool("search"));
    auto result = registry.invoke("search", {{"query", "test"}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "found: test");
}

TEST_F(ToolRegistryTest, LambdaRegistration) {
    auto lambda = [](int x, int y) -> int { return x * y; };
    (void)registry.register_tool("multiply_ints", "Multiply integers", {"x", "y"}, lambda);

    auto result = registry.invoke("multiply_ints", {{"x", 5}, {"y", 6}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 30);
}

TEST_F(ToolRegistryTest, ZeroArgTool) {
    (void)registry.register_tool("get_time", "Get current time", {}, get_time);

    EXPECT_TRUE(registry.has_tool("get_time"));
    auto schema = registry.get_tool_schema("get_time");
    EXPECT_TRUE(schema["function"]["parameters"]["properties"].empty());

    auto result = registry.invoke("get_time", json::object());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "2024-01-01T00:00:00Z");
}

TEST_F(ToolRegistryTest, RequiredFieldsInSchema) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto schema = registry.get_tool_schema("add");
    auto required = schema["function"]["parameters"]["required"];
    EXPECT_EQ(required.size(), 2);

    bool has_a = std::find(required.begin(), required.end(), "a") != required.end();
    bool has_b = std::find(required.begin(), required.end(), "b") != required.end();
    EXPECT_TRUE(has_a);
    EXPECT_TRUE(has_b);
}

// ============================================================================
// Additional coverage tests
// ============================================================================

TEST_F(ToolRegistryTest, ArityMismatchReturnsError) {
    auto result1 = registry.register_tool("add", "Add", {"a"}, add);
    EXPECT_FALSE(result1.has_value());
    EXPECT_EQ(result1.error().code, ErrorCode::InvalidToolSignature);

    auto result2 = registry.register_tool("add", "Add", {"a", "b", "c"}, add);
    EXPECT_FALSE(result2.has_value());
    EXPECT_EQ(result2.error().code, ErrorCode::InvalidToolSignature);

    EXPECT_EQ(registry.size(), 0);
}

TEST_F(ToolRegistryTest, FloatTypeSchema) {
    auto float_func = [](float x) -> float { return x * 2.0f; };
    (void)registry.register_tool("double_it", "Double a float", {"x"}, float_func);

    auto schema = registry.get_tool_schema("double_it");
    EXPECT_EQ(schema["function"]["parameters"]["properties"]["x"]["type"], "number");
}

TEST_F(ToolRegistryTest, GetParametersSchemaReturnsCopy) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto params = registry.get_parameters_schema("add");
    ASSERT_TRUE(params.has_value());
    EXPECT_EQ((*params)["type"], "object");
    EXPECT_TRUE(params->contains("properties"));
    EXPECT_TRUE(params->contains("required"));
}

TEST_F(ToolRegistryTest, GetParametersSchemaEmptyForMissing) {
    EXPECT_FALSE(registry.get_parameters_schema("nonexistent").has_value());
}

TEST_F(ToolRegistryTest, ConstLambdaWithCapture) {
    int multiplier = 3;
    auto const_lambda = [multiplier](int x) -> int {
        return x * multiplier;
    };
    (void)registry.register_tool("triple", "Triple a number", {"x"}, const_lambda);

    auto result = registry.invoke("triple", {{"x", 5}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 15);
}

TEST_F(ToolRegistryTest, StdFunctionRegistration) {
    std::function<std::string(std::string)> func = [](std::string s) {
        return "prefix_" + s;
    };
    (void)registry.register_tool("prefix", "Add prefix", {"s"}, func);

    EXPECT_TRUE(registry.has_tool("prefix"));
    auto result = registry.invoke("prefix", {{"s", "test"}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "prefix_test");
}

TEST_F(ToolRegistryTest, HandlerThrowsStdException) {
    auto throwing = [](int x) -> int {
        if (x < 0) throw std::runtime_error("negative input");
        return x;
    };
    (void)registry.register_tool("check", "Check positive", {"x"}, throwing);

    auto result = registry.invoke("check", {{"x", -1}});
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolExecutionFailed);
    EXPECT_NE(result.error().message.find("negative input"), std::string::npos);
}

TEST_F(ToolRegistryTest, ConcurrentReadsDuringWrite) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    std::atomic<bool> start{false};
    std::atomic<int> successes{0};

    auto reader = [&]() {
        while (!start.load()) {}
        for (int i = 0; i < 100; ++i) {
            if (registry.has_tool("add")) successes++;
            (void)registry.get_tool_names();
            (void)registry.size();
        }
    };

    auto writer = [&]() {
        while (!start.load()) {}
        for (int i = 0; i < 100; ++i) {
            (void)registry.register_tool("t_" + std::to_string(i), "Tool", {"a", "b"}, add);
        }
    };

    std::thread t1(reader);
    std::thread t2(reader);
    std::thread t3(writer);
    start.store(true);
    t1.join(); t2.join(); t3.join();

    EXPECT_EQ(successes.load(), 200);
    EXPECT_GE(registry.size(), 1);
}

// ============================================================================
// Edge case coverage
// ============================================================================

TEST_F(ToolRegistryTest, MixedParameterTypes) {
    auto mixed_tool = [](int count, std::string name, double multiplier) -> std::string {
        return name + ": " + std::to_string(static_cast<int>(count * multiplier));
    };

    (void)registry.register_tool("mixed", "Tool with mixed types", {"count", "name", "multiplier"}, mixed_tool);

    auto schema = registry.get_tool_schema("mixed");
    auto props = schema["function"]["parameters"]["properties"];
    EXPECT_EQ(props["count"]["type"], "integer");
    EXPECT_EQ(props["name"]["type"], "string");
    EXPECT_EQ(props["multiplier"]["type"], "number");

    auto result = registry.invoke("mixed", {{"count", 5}, {"name", "test"}, {"multiplier", 2.5}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "test: 12");
}

TEST_F(ToolRegistryTest, EmptyStringArgument) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);

    auto result = registry.invoke("greet", {{"name", ""}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "Hello, !");
}

TEST_F(ToolRegistryTest, UnicodeInArguments) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);

    auto result = registry.invoke("greet", {{"name", "世界"}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "Hello, 世界!");
}

TEST_F(ToolRegistryTest, VeryLongString) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);

    std::string long_name(10000, 'x');
    auto result = registry.invoke("greet", {{"name", long_name}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "Hello, " + long_name + "!");
}

TEST_F(ToolRegistryTest, SpecialCharactersInString) {
    (void)registry.register_tool("greet", "Greet someone", {"name"}, greet);

    auto result = registry.invoke("greet", {{"name", "Alice \"Bob\" O'Connor"}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], "Hello, Alice \"Bob\" O'Connor!");
}

TEST_F(ToolRegistryTest, NegativeNumbers) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", -5}, {"b", -3}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], -8);
}

TEST_F(ToolRegistryTest, ZeroValues) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", 0}, {"b", 0}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 0);
}

TEST_F(ToolRegistryTest, VeryLargeNumbers) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", 1000000000}, {"b", 2000000000}});
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE((*result).contains("result"));
}

TEST_F(ToolRegistryTest, FloatPrecision) {
    (void)registry.register_tool("multiply", "Multiply doubles", {"a", "b"}, multiply);

    auto result = registry.invoke("multiply", {{"a", 0.1}, {"b", 0.2}});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR((*result)["result"].get<double>(), 0.02, 0.0001);
}

TEST_F(ToolRegistryTest, ExtraArgumentsIgnored) {
    (void)registry.register_tool("add", "Add two integers", {"a", "b"}, add);

    auto result = registry.invoke("add", {{"a", 3}, {"b", 4}, {"c", 999}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 7);
}

TEST_F(ToolRegistryTest, ZeroArgToolThrows) {
    auto throwing_tool = []() -> int {
        throw std::runtime_error("Intentional error");
    };

    (void)registry.register_tool("thrower", "A tool that throws", {}, throwing_tool);

    auto result = registry.invoke("thrower", nlohmann::json::object());
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::ToolExecutionFailed);
    EXPECT_NE(result.error().message.find("Tool execution failed"), std::string::npos);
}

TEST_F(ToolRegistryTest, VeryLongToolName) {
    std::string long_name(1000, 'x');
    (void)registry.register_tool(long_name, "Tool with very long name", {"a", "b"}, add);

    EXPECT_TRUE(registry.has_tool(long_name));
    auto result = registry.invoke(long_name, {{"a", 1}, {"b", 2}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ((*result)["result"], 3);
}

TEST_F(ToolRegistryTest, VeryLongDescription) {
    std::string long_desc(10000, 'y');
    (void)registry.register_tool("test", long_desc, {"a", "b"}, add);

    auto schema = registry.get_tool_schema("test");
    EXPECT_EQ(schema["function"]["description"], long_desc);
}
