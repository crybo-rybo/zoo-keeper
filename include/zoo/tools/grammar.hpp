#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace zoo::tools {

class GrammarBuilder {
public:
    static std::string build(const nlohmann::json& tool_schemas) {
        if (!tool_schemas.is_array() || tool_schemas.empty()) return {};

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
    static std::string sanitize(const std::string& name) {
        std::string result;
        for (char c : name) {
            result += std::isalnum(static_cast<unsigned char>(c)) ? c : '-';
        }
        return result;
    }

    static std::string type_rule(const std::string& json_type) {
        if (json_type == "integer") return "integer";
        if (json_type == "number") return "number";
        if (json_type == "string") return "string";
        if (json_type == "boolean") return "boolean";
        return "string";
    }

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
