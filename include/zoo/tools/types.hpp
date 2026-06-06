/**
 * @file types.hpp
 * @brief Shared tool-calling value types.
 */

#pragma once

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

} // namespace zoo::tools
