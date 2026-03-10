/**
 * @file validation.hpp
 * @brief Tool-call validation helpers.
 */

#pragma once

#include "registry.hpp"
#include "types.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace zoo::tools {

/**
 * @brief Validates parsed tool calls against normalized registry metadata.
 */
class ToolArgumentsValidator {
  public:
    /**
     * @brief Validates one parsed tool call against the registered schema.
     *
     * @param tool_call Parsed tool call to inspect.
     * @param registry Registry used to look up the normalized tool metadata.
     * @return Empty success when the arguments satisfy the registered schema.
     */
    Expected<void> validate(const ToolCall& tool_call, const ToolRegistry& registry) const {
        auto metadata = registry.get_tool_metadata(tool_call.name);
        if (!metadata) {
            return std::unexpected(
                Error{ErrorCode::ToolNotFound, "Tool not found: " + tool_call.name});
        }

        return validate(tool_call, *metadata);
    }

    /**
     * @brief Validates one parsed tool call against supplied metadata.
     *
     * @param tool_call Parsed tool call to inspect.
     * @param metadata Normalized metadata for the named tool.
     * @return Empty success when the arguments satisfy the registered schema.
     */
    Expected<void> validate(const ToolCall& tool_call, const ToolMetadata& metadata) const {
        if (!tool_call.arguments.is_object()) {
            return std::unexpected(
                Error{ErrorCode::ToolValidationFailed, "Tool arguments must be a JSON object"});
        }

        std::unordered_map<std::string_view, const ToolParameter*> parameters_by_name;
        parameters_by_name.reserve(metadata.parameters.size());
        for (const auto& parameter : metadata.parameters) {
            parameters_by_name.emplace(parameter.name, &parameter);
        }

        for (const auto& parameter : metadata.parameters) {
            if (parameter.required && !tool_call.arguments.contains(parameter.name)) {
                return std::unexpected(Error{ErrorCode::ToolValidationFailed,
                                             "Missing required argument: " + parameter.name});
            }
        }

        for (const auto& [key, value] : tool_call.arguments.items()) {
            auto it = parameters_by_name.find(key);
            if (it == parameters_by_name.end()) {
                return std::unexpected(
                    Error{ErrorCode::ToolValidationFailed, "Unexpected argument: " + key});
            }

            const ToolParameter& parameter = *it->second;
            if (!detail::json_matches_type(value, parameter.type)) {
                return std::unexpected(Error{ErrorCode::ToolValidationFailed,
                                             "Argument '" + key + "' has wrong type: expected " +
                                                 std::string(tool_value_type_name(parameter.type)) +
                                                 ", got " + json_type_name(value)});
            }

            if (!parameter.enum_values.empty() && !matches_enum(value, parameter.enum_values)) {
                return std::unexpected(
                    Error{ErrorCode::ToolValidationFailed,
                          "Argument '" + key + "' must match one of the registered enum values"});
            }
        }

        return {};
    }

  private:
    static bool matches_enum(const nlohmann::json& value,
                             const std::vector<nlohmann::json>& enum_values) {
        for (const auto& enum_value : enum_values) {
            if (enum_value == value) {
                return true;
            }
        }
        return false;
    }

    static const char* json_type_name(const nlohmann::json& val) {
        if (val.is_null()) {
            return "null";
        }
        if (val.is_boolean()) {
            return "boolean";
        }
        if (val.is_number_integer()) {
            return "integer";
        }
        if (val.is_number_float()) {
            return "number";
        }
        if (val.is_string()) {
            return "string";
        }
        if (val.is_array()) {
            return "array";
        }
        if (val.is_object()) {
            return "object";
        }
        return "unknown";
    }
};

} // namespace zoo::tools
