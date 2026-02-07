#pragma once

#include "../types.hpp"
#include "tool_registry.hpp"
#include "tool_call_parser.hpp"
#include <nlohmann/json.hpp>
#include <string>
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
        if (!registry.has_tool(tool_call.name)) {
            return "Tool not found: " + tool_call.name;
        }

        auto schema = registry.get_tool_schema(tool_call.name);
        auto params = schema["function"]["parameters"];

        // Check required fields
        if (params.contains("required")) {
            for (const auto& req : params["required"]) {
                std::string field = req.get<std::string>();
                if (!tool_call.arguments.contains(field)) {
                    return "Missing required argument: " + field;
                }
            }
        }

        // Check types of provided arguments
        if (params.contains("properties")) {
            for (auto& [key, prop] : params["properties"].items()) {
                if (tool_call.arguments.contains(key)) {
                    std::string expected_type = prop["type"].get<std::string>();
                    const auto& val = tool_call.arguments[key];

                    if (!type_matches(val, expected_type)) {
                        return "Argument '" + key + "' has wrong type: expected " +
                               expected_type + ", got " + json_type_name(val);
                    }
                }
            }
        }

        return "";  // Valid
    }

    /**
     * Build an error message to inject into conversation as a system message.
     */
    static Message build_error_message(const std::string& tool_name,
                                       const std::string& error) {
        std::string content = "Tool call error for '" + tool_name + "': " + error +
                              "\nPlease correct the arguments and try again.";
        return Message::system(std::move(content));
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

    static bool type_matches(const nlohmann::json& val, const std::string& expected) {
        if (expected == "integer") return val.is_number_integer();
        if (expected == "number") return val.is_number();
        if (expected == "string") return val.is_string();
        if (expected == "boolean") return val.is_boolean();
        if (expected == "object") return val.is_object();
        if (expected == "array") return val.is_array();
        return false;
    }

    static std::string json_type_name(const nlohmann::json& val) {
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
