#pragma once

#include "../types.hpp"
#include "tool_registry.hpp"
#include "tool_call_parser.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>

namespace zoo {
namespace engine {

/**
 * @brief Validates tool call arguments against registered schemas and tracks retries.
 *
 * The ErrorRecovery class provides argument validation for tool calls by comparing
 * the provided arguments against the registered JSON schema. It also maintains retry
 * counts per tool to enforce maximum retry limits for failed tool calls.
 *
 * @threadsafety Not thread-safe. Intended for use on a single inference thread.
 */
class ErrorRecovery {
public:
    /**
     * @brief Construct an ErrorRecovery instance with a maximum retry limit.
     *
     * @param max_retries Maximum number of retry attempts per tool (default: 2)
     */
    explicit ErrorRecovery(int max_retries = 2)
        : max_retries_(max_retries)
    {}

    /**
     * @brief Validate tool call arguments against the registered schema.
     *
     * Performs comprehensive validation including:
     * - Checking that the tool exists in the registry
     * - Verifying all required arguments are present
     * - Validating argument types match the schema
     *
     * @param tool_call The tool call to validate
     * @param registry The tool registry containing schema definitions
     * @return Empty string if valid, error message describing the validation failure otherwise
     */
    std::string validate_args(const ToolCall& tool_call, const ToolRegistry& registry) const {
        // get_parameters_schema() returns a copy so validation proceeds without
        // holding the registry lock, avoiding a use-after-free if a concurrent
        // register_tool() reallocates the internal map.
        auto params_opt = registry.get_parameters_schema(tool_call.name);
        if (!params_opt) {
            return "Tool not found: " + tool_call.name;
        }
        const nlohmann::json& params = *params_opt;

        // Check required fields
        if (params.contains("required")) {
            for (const auto& req : params["required"]) {
                // Use get_ref to avoid allocating a new std::string per field
                const auto& field = req.get_ref<const std::string&>();
                if (!tool_call.arguments.contains(field)) {
                    return "Missing required argument: " + field;
                }
            }
        }

        // Check types of provided arguments
        auto props_it = params.find("properties");
        if (props_it != params.end()) {
            for (const auto& [key, prop] : props_it->items()) {
                // Single lookup via find() instead of contains() + operator[]
                auto arg_it = tool_call.arguments.find(key);
                if (arg_it != tool_call.arguments.end()) {
                    auto type_it = prop.find("type");
                    if (type_it == prop.end() || !type_it->is_string()) {
                        continue;  // Skip type check if property schema lacks type info
                    }
                    const auto& expected_type = type_it->get_ref<const std::string&>();

                    if (!type_matches(*arg_it, expected_type)) {
                        return "Argument '" + key + "' has wrong type: expected " +
                               expected_type + ", got " + json_type_name(*arg_it);
                    }
                }
            }
        }

        return "";  // Valid
    }

    /**
     * @brief Check if we can retry the tool call.
     *
     * Returns true if the tool has not exceeded the maximum retry limit.
     *
     * @param tool_name Name of the tool to check
     * @return True if the tool can be retried, false if retry limit reached
     */
    bool can_retry(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return true;
        return it->second < max_retries_;
    }

    /**
     * @brief Record a retry attempt for a tool.
     *
     * Increments the retry counter for the specified tool. Call this after
     * a tool call validation failure or execution error.
     *
     * @param tool_name Name of the tool to record a retry for
     */
    void record_retry(const std::string& tool_name) {
        retry_counts_[tool_name]++;
    }

    /**
     * @brief Get current retry count for a tool.
     *
     * Returns the number of retry attempts that have been recorded for the
     * specified tool during the current request processing.
     *
     * @param tool_name Name of the tool to query
     * @return Number of retry attempts (0 if no retries recorded)
     */
    int get_retry_count(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return 0;
        return it->second;
    }

    /**
     * @brief Reset retry tracking for all tools.
     *
     * Clears all retry counters. This should be called between requests to
     * ensure each new request starts with a clean retry state.
     */
    void reset() {
        retry_counts_.clear();
    }

    /**
     * @brief Get the configured maximum retry limit.
     *
     * @return Maximum number of retry attempts allowed per tool
     */
    int max_retries() const { return max_retries_; }

private:
    int max_retries_;
    std::unordered_map<std::string, int> retry_counts_;

    /**
     * @brief Check if a JSON value matches the expected JSON Schema type.
     *
     * @param val The JSON value to check
     * @param expected The expected JSON Schema type name (e.g., "integer", "string")
     * @return True if the value's type matches the expected type
     */
    static bool type_matches(const nlohmann::json& val, std::string_view expected) {
        if (expected == "integer") return val.is_number_integer();
        if (expected == "number") return val.is_number();
        if (expected == "string") return val.is_string();
        if (expected == "boolean") return val.is_boolean();
        if (expected == "object") return val.is_object();
        if (expected == "array") return val.is_array();
        return false;
    }

    /**
     * @brief Get a human-readable type name for a JSON value.
     *
     * @param val The JSON value
     * @return String representation of the value's type
     */
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

} // namespace engine
} // namespace zoo
