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
 * Validates tool call arguments against registered schemas and tracks retries.
 */
class ErrorRecovery {
public:
    explicit ErrorRecovery(int max_retries = 2)
        : max_retries_(max_retries)
    {}

    /**
     * Validate tool call arguments against the registered schema.
     *
     * @return Empty string if valid, error message if invalid.
     */
    std::string validate_args(const ToolCall& tool_call, const ToolRegistry& registry) const {
        // Single lookup: get_parameters_schema() does one lock + one find(),
        // instead of has_tool() + get_tool_schema() (two locks + two finds + wrapper JSON).
        const auto* params = registry.get_parameters_schema(tool_call.name);
        if (!params) {
            return "Tool not found: " + tool_call.name;
        }

        // Check required fields
        if (params->contains("required")) {
            for (const auto& req : (*params)["required"]) {
                // Use get_ref to avoid allocating a new std::string per field
                const auto& field = req.get_ref<const std::string&>();
                if (!tool_call.arguments.contains(field)) {
                    return "Missing required argument: " + field;
                }
            }
        }

        // Check types of provided arguments
        auto props_it = params->find("properties");
        if (props_it != params->end()) {
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
     * Check if we can retry the tool call.
     */
    bool can_retry(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return true;
        return it->second < max_retries_;
    }

    /**
     * Record a retry attempt.
     */
    void record_retry(const std::string& tool_name) {
        retry_counts_[tool_name]++;
    }

    /**
     * Get current retry count for a tool.
     */
    int get_retry_count(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return 0;
        return it->second;
    }

    /**
     * Reset retry tracking (between requests).
     */
    void reset() {
        retry_counts_.clear();
    }

    int max_retries() const { return max_retries_; }

private:
    int max_retries_;
    std::unordered_map<std::string, int> retry_counts_;

    static bool type_matches(const nlohmann::json& val, std::string_view expected) {
        if (expected == "integer") return val.is_number_integer();
        if (expected == "number") return val.is_number();
        if (expected == "string") return val.is_string();
        if (expected == "boolean") return val.is_boolean();
        if (expected == "object") return val.is_object();
        if (expected == "array") return val.is_array();
        return false;
    }

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
