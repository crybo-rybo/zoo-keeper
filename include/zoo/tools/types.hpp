/**
 * @file types.hpp
 * @brief Shared tool-calling value types and handler aliases.
 */

#pragma once

#include <atomic>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
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
 * @brief Metadata and executable handler for a registered tool.
 */
struct ToolEntry {
    std::string name;                 ///< Public tool name presented to the model.
    std::string description;          ///< Human-readable tool description for prompts and schemas.
    nlohmann::json parameters_schema; ///< JSON Schema describing accepted arguments.
    ToolHandler handler;              ///< Callable invoked when the tool is executed.
};

} // namespace zoo::tools
