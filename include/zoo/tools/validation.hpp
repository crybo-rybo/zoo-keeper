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

        for (const auto& parameter : metadata.parameters) {
            if (parameter.required && !tool_call.arguments.contains(parameter.name)) {
                return std::unexpected(Error{ErrorCode::ToolValidationFailed,
                                             "Missing required argument: " + parameter.name});
            }
        }

        for (const auto& [key, value] : tool_call.arguments.items()) {
            const ToolParameter* parameter = find_parameter(metadata.parameters, key);
            if (!parameter) {
                return std::unexpected(
                    Error{ErrorCode::ToolValidationFailed, "Unexpected argument: " + key});
            }

            if (!detail::json_matches_type(value, parameter->type)) {
                return std::unexpected(
                    Error{ErrorCode::ToolValidationFailed,
                          "Argument '" + key + "' has wrong type: expected " +
                              std::string(tool_value_type_name(parameter->type)) + ", got " +
                              json_type_name(value)});
            }

            if (!parameter->enum_values.empty() && !matches_enum(value, parameter->enum_values)) {
                return std::unexpected(
                    Error{ErrorCode::ToolValidationFailed,
                          "Argument '" + key + "' must match one of the registered enum values"});
            }
        }

        return {};
    }

  private:
    static const ToolParameter* find_parameter(const std::vector<ToolParameter>& parameters,
                                               std::string_view name) {
        for (const auto& parameter : parameters) {
            if (parameter.name == name) {
                return &parameter;
            }
        }
        return nullptr;
    }

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

/**
 * @brief Validates a JSON object against a parameter vector without needing a ToolCall or registry.
 *
 * @param data JSON object to validate.
 * @param parameters Normalized parameter schema to validate against.
 * @return Empty success when the data satisfies the schema, or a
 *         `ToolValidationFailed` error identifying the first constraint violation.
 */
[[nodiscard]] inline Expected<void>
validate_json_against_schema(const nlohmann::json& data,
                             const std::vector<ToolParameter>& parameters) {
    if (!data.is_object()) {
        return std::unexpected(
            Error{ErrorCode::ToolValidationFailed, "Data must be a JSON object"});
    }

    for (const auto& parameter : parameters) {
        if (parameter.required && !data.contains(parameter.name)) {
            return std::unexpected(Error{ErrorCode::ToolValidationFailed,
                                         "Missing required field: " + parameter.name});
        }
    }

    for (const auto& [key, value] : data.items()) {
        const ToolParameter* found = nullptr;
        for (const auto& parameter : parameters) {
            if (parameter.name == key) {
                found = &parameter;
                break;
            }
        }

        if (!found) {
            return std::unexpected(
                Error{ErrorCode::ToolValidationFailed, "Unexpected field: " + key});
        }

        if (!detail::json_matches_type(value, found->type)) {
            return std::unexpected(
                Error{ErrorCode::ToolValidationFailed,
                      "Field '" + key + "' has wrong type: expected " +
                          std::string(tool_value_type_name(found->type))});
        }

        if (!found->enum_values.empty()) {
            bool matched = false;
            for (const auto& enum_value : found->enum_values) {
                if (enum_value == value) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                return std::unexpected(
                    Error{ErrorCode::ToolValidationFailed,
                          "Field '" + key + "' must match one of the registered enum values"});
            }
        }
    }

    return {};
}

} // namespace zoo::tools
