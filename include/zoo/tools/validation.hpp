#pragma once

#include "types.hpp"
#include "registry.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>

namespace zoo::tools {

class ErrorRecovery {
public:
    explicit ErrorRecovery(int max_retries = 2)
        : max_retries_(max_retries)
    {}

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

    bool can_retry(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return true;
        return it->second < max_retries_;
    }

    void record_retry(const std::string& tool_name) {
        retry_counts_[tool_name]++;
    }

    int get_retry_count(const std::string& tool_name) const {
        auto it = retry_counts_.find(tool_name);
        if (it == retry_counts_.end()) return 0;
        return it->second;
    }

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

} // namespace zoo::tools
