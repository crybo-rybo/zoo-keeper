/**
 * @file grammar.hpp
 * @brief Grammar generation helpers for constrained tool calling.
 */

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>

namespace zoo::tools {

/**
 * @brief Builds a GBNF grammar that constrains output to registered tool calls.
 */
class GrammarBuilder {
public:
    /**
     * @brief Builds a grammar from the registry's tool schemas.
     *
     * @param tool_schemas Array returned by `ToolRegistry::get_all_schemas()`.
     * @return A GBNF grammar rooted at `root`, or an empty string when the
     *         schemas are invalid or cannot be represented safely.
     */
    static std::string build(const nlohmann::json& tool_schemas) {
        if (!tool_schemas.is_array() || tool_schemas.empty()) return {};

        // Reject tool name collisions after sanitization
        std::unordered_set<std::string> seen_names;
        for (const auto& schema : tool_schemas) {
            auto safe = sanitize(schema["function"]["name"].get<std::string>());
            if (!seen_names.insert(safe).second) return {};
        }

        std::string grammar;
        grammar += "root ::= \"<tool_call>\" ws tool-call ws \"</tool_call>\"\n";

        // tool-call ::= tool-add | tool-multiply | ...
        grammar += "tool-call ::= ";
        for (size_t i = 0; i < tool_schemas.size(); ++i) {
            if (i > 0) grammar += " | ";
            grammar += "tool-" + sanitize(tool_schemas[i]["function"]["name"].get<std::string>());
        }
        grammar += "\n";

        // Per-tool rules
        for (const auto& schema : tool_schemas) {
            const auto& func = schema["function"];
            const std::string name = func["name"].get<std::string>();
            const std::string safe = sanitize(name);
            const auto& params = func["parameters"];

            grammar += "tool-" + safe + " ::= \"{\" ws "
                "\"\\\"name\\\"\" ws \":\" ws \"\\\"" + name + "\\\"\" ws "
                "\",\" ws \"\\\"arguments\\\"\" ws \":\" ws \"{\" ws ";

            if (params.contains("required") && !params["required"].empty()) {
                grammar += safe + "-args ws ";
            }

            grammar += "\"}\" ws \"}\"\n";

            // Args rule
            if (params.contains("required") && !params["required"].empty()) {
                grammar += safe + "-args ::= ";
                const auto& props = params["properties"];
                const auto& required = params["required"];

                for (size_t i = 0; i < required.size(); ++i) {
                    if (i > 0) grammar += " ws \",\" ws ";
                    const std::string arg_name = required[i].get<std::string>();
                    const std::string arg_type = props[arg_name]["type"].get<std::string>();
                    grammar += "\"\\\"" + arg_name + "\\\"\" ws \":\" ws " + type_rule(arg_type);
                }
                grammar += "\n";
            }
        }

        grammar += primitive_rules();
        return grammar;
    }

private:
    /// Converts tool names into grammar-safe rule identifiers.
    static std::string sanitize(const std::string& name) {
        std::string result;
        for (char c : name) {
            result += std::isalnum(static_cast<unsigned char>(c)) ? c : '-';
        }
        return result;
    }

    /// Maps a JSON Schema primitive type to the corresponding grammar rule name.
    static std::string type_rule(const std::string& json_type) {
        if (json_type == "integer") return "integer";
        if (json_type == "number") return "number";
        if (json_type == "string") return "string";
        if (json_type == "boolean") return "boolean";
        return "string";
    }

    /// Returns the shared primitive grammar rules appended to every generated grammar.
    static std::string primitive_rules() {
        return
            "integer ::= \"-\"? [0-9]+\n"
            "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n"
            "string ::= \"\\\"\" [^\"\\\\]* (\"\\\\\" [^\\x00] [^\"\\\\]*)* \"\\\"\"\n"
            "boolean ::= \"true\" | \"false\"\n"
            "ws ::= [ \\t\\n]*\n";
    }
};

} // namespace zoo::tools
