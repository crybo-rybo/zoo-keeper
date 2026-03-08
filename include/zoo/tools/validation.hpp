/**
 * @file validation.hpp
 * @brief Tool-call validation and retry bookkeeping utilities.
 */

#pragma once

#include "types.hpp"
#include "registry.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>

namespace zoo::tools {

/**
 * @brief Validates parsed tool calls against registry metadata and tracks retries.
 */
class ErrorRecovery {
public:
    /**
     * @brief Creates a validator with a fixed retry budget per tool name.
     *
     * @param max_retries Maximum number of retries allowed for a malformed tool call.
     */
    explicit ErrorRecovery(int max_retries = 2)
        : max_retries_(max_retries)
    {}

    /**
     * @brief Validates a tool call against the registered parameter schema.
     *
     * @param tool_call Parsed tool call to inspect.
     * @param registry Registry used to look up the tool schema.
     * @return Empty string on success, or a human-readable validation error.
     */
    std::string validate_args(const ToolCall& tool_call, const ToolRegistry& registry) const {
        auto params_opt = registry.get_parameters_schema(tool_call.name);
        if (!params_opt) {
            return "Tool not found: " + tool_call.name;
        }
        const nlohmann::json& params = *params_opt;

        if (params.contains("required")) {
            for (const auto& req : params["required"]) {
                const auto& field = req.get_ref<const std::string&>();
                if (!tool_call.arguments.contains(field)) {
                    return "Missing required argument: " + field;
                }
            }
        }

        auto props_it = params.find("properties");
        if (props_it != params.end()) {
            for (const auto& [key, prop] : props_it->items()) {
                auto arg_it = tool_call.arguments.find(key);
                if (arg_it != tool_call.arguments.end()) {
                    auto type_it = prop.find("type");
                    if (type_it == prop.end() || !type_it->is_string()) {
                        continue;
                    }
                    const auto& expected_type = type_it->get_ref<const std::string&>();
                    if (!type_matches(*arg_it, expected_type)) {
                        return "Argument '" + key + "' has wrong type: expected " +
                               expected_type + ", got " + json_type_name(*arg_it);
                    }
                }
            }
        }

        return "";
    }

    /**
     * @brief Reports whether another retry is allowed for `tool_name`.
     *
     * @param tool_name Tool identifier used as the retry bucket.
     * @return `true` when the recorded retry count is still below the limit.
     */
    bool can_retry(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return true;
        return it->second < max_retries_;
    }

    /**
     * @brief Records one additional retry for `tool_name`.
     *
     * @param tool_name Tool identifier whose retry count should be incremented.
     */
    void record_retry(const std::string& tool_name) {
        retry_counts_[tool_name]++;
    }

    /**
     * @brief Returns the number of recorded retries for `tool_name`.
     *
     * @param tool_name Tool identifier whose retry count should be queried.
     * @return Current retry count, or zero if none have been recorded.
     */
    int get_retry_count(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return 0;
        return it->second;
    }

    /// Clears all recorded retry counts.
    void reset() {
        retry_counts_.clear();
    }

    /// Returns the configured maximum retry count.
    int max_retries() const { return max_retries_; }

private:
    int max_retries_;
    std::unordered_map<std::string, int> retry_counts_;

    /// Returns whether a JSON value matches the expected JSON Schema primitive type.
    static bool type_matches(const nlohmann::json& val, std::string_view expected) {
        if (expected == "integer") return val.is_number_integer();
        if (expected == "number") return val.is_number();
        if (expected == "string") return val.is_string();
        if (expected == "boolean") return val.is_boolean();
        if (expected == "object") return val.is_object();
        if (expected == "array") return val.is_array();
        return false;
    }

    /// Returns a stable human-readable name for a JSON value type.
    static const char* json_type_name(const nlohmann::json& val) {
        if (val.is_null()) return "null";
        if (val.is_boolean()) return "boolean";
        if (val.is_number_integer()) return "integer";
        if (val.is_number_float()) return "number";
        if (val.is_string()) return "string";
        if (val.is_array()) return "array";
        if (val.is_object()) return "object";
        return "unknown";
    }
};

} // namespace zoo::tools
