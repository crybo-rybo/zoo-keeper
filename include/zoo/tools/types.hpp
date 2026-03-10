/**
 * @file types.hpp
 * @brief Shared tool-calling value types and handler aliases.
 */

#pragma once

#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <zoo/core/types.hpp>

namespace zoo::tools {

/**
 * @brief Structured tool call extracted from model output.
 */
struct ToolCall {
    std::string id;           ///< Unique identifier for correlating tool responses.
    std::string name;         ///< Registered tool name to invoke.
    nlohmann::json arguments; ///< JSON arguments supplied by the model.

    /// Compares two tool calls field-by-field.
    bool operator==(const ToolCall& other) const = default;
};

/**
 * @brief Signature of a tool handler stored in the registry.
 *
 * Handlers receive the JSON argument object and return either a JSON payload or
 * a `zoo::Error`.
 */
using ToolHandler = std::function<Expected<nlohmann::json>(const nlohmann::json&)>;

/**
 * @brief Supported primitive value kinds for registered tool parameters.
 */
enum class ToolValueType {
    Integer,
    Number,
    String,
    Boolean,
};

/// Returns the JSON Schema primitive string for a supported tool value type.
[[nodiscard]] inline const char* tool_value_type_name(ToolValueType type) noexcept {
    switch (type) {
    case ToolValueType::Integer:
        return "integer";
    case ToolValueType::Number:
        return "number";
    case ToolValueType::String:
        return "string";
    case ToolValueType::Boolean:
        return "boolean";
    }
    return "unknown";
}

/**
 * @brief Normalized schema metadata for one tool parameter.
 */
struct ToolParameter {
    std::string name;                           ///< Public parameter name.
    ToolValueType type = ToolValueType::String; ///< Supported primitive type.
    bool required = false;   ///< Whether the parameter must be present in arguments.
    std::string description; ///< Optional human-readable parameter description.
    std::vector<nlohmann::json>
        enum_values; ///< Optional enum domain, expressed as exact JSON literals.

    /// Compares two parameter schemas field-by-field.
    bool operator==(const ToolParameter& other) const = default;
};

/**
 * @brief Public metadata stored for one registered tool.
 */
struct ToolMetadata {
    std::string name;                 ///< Public tool name presented to the model.
    std::string description;          ///< Human-readable tool description for prompts and schemas.
    nlohmann::json parameters_schema; ///< Normalized JSON Schema describing accepted arguments.
    std::vector<ToolParameter>
        parameters; ///< Canonical parameter order used by validation/grammar.

    /// Compares two tool metadata records field-by-field.
    bool operator==(const ToolMetadata& other) const = default;
};

/**
 * @brief Metadata plus executable handler for a registered tool.
 */
struct ToolDefinition {
    ToolMetadata metadata; ///< Public metadata exposed to prompts and validators.
    ToolHandler handler;   ///< Callable invoked when the tool is executed.
};

} // namespace zoo::tools
