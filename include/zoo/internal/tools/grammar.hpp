/**
 * @file grammar.hpp
 * @brief Grammar generation helpers for constrained tool calling.
 */

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <zoo/tools/types.hpp>

namespace zoo::tools {

/**
 * @brief Builds a GBNF grammar that constrains output to registered tool calls.
 */
class GrammarBuilder {
  public:
    /**
     * @brief Builds a grammar from normalized tool metadata.
     *
     * @param tools Registered tools in canonical registration order.
     * @return A GBNF grammar rooted at `root`, or an empty string when no tools
     *         are registered.
     */
    static std::string build(const std::vector<ToolMetadata>& tools) {
        if (tools.empty()) {
            return {};
        }

        std::string grammar;
        grammar += "root ::= " + literal("<tool_call>") + " ws tool-call ws " +
                   literal("</tool_call>") + "\n";
        grammar += "tool-call ::= ";
        for (size_t i = 0; i < tools.size(); ++i) {
            if (i > 0) {
                grammar += " | ";
            }
            grammar += tool_rule_name(i);
        }
        grammar += "\n";

        for (size_t tool_index = 0; tool_index < tools.size(); ++tool_index) {
            append_tool_rules(grammar, tool_index, tools[tool_index]);
        }

        grammar += primitive_rules();
        return grammar;
    }

  private:
    static void append_tool_rules(std::string& grammar, size_t tool_index, const ToolMetadata& tool) {
        grammar += tool_rule_name(tool_index) + " ::= " + literal("{") + " ws " +
                   json_string_literal("name") + " ws " + literal(":") + " ws " +
                   json_string_literal(tool.name) + " ws " + literal(",") + " ws " +
                   json_string_literal("arguments") + " ws " + literal(":") + " ws " +
                   literal("{") + " ws ";

        if (!tool.parameters.empty()) {
            grammar += args_rule_name(tool_index) + " ws ";
        }

        grammar += literal("}") + " ws " + literal("}") + "\n";

        if (tool.parameters.empty()) {
            return;
        }

        const size_t required_count = count_required_prefix(tool.parameters);
        append_parameter_rules(grammar, tool_index, tool);

        grammar += args_rule_name(tool_index) + " ::= ";
        if (required_count == tool.parameters.size()) {
            grammar += build_required_sequence(tool_index, 0, required_count);
        } else if (required_count == 0) {
            grammar += start_rule_name(tool_index, 0);
        } else {
            grammar += build_required_sequence(tool_index, 0, required_count) + " " +
                       cont_rule_name(tool_index, required_count);
        }
        grammar += "\n";

        if (required_count < tool.parameters.size()) {
            append_optional_rules(grammar, tool_index, tool.parameters, required_count);
        }
    }

    static void append_parameter_rules(std::string& grammar, size_t tool_index,
                                       const ToolMetadata& tool) {
        for (size_t param_index = 0; param_index < tool.parameters.size(); ++param_index) {
            const auto& parameter = tool.parameters[param_index];
            grammar += param_rule_name(tool_index, param_index) + " ::= " +
                       json_string_literal(parameter.name) + " ws " + literal(":") + " ws ";

            if (parameter.enum_values.empty()) {
                grammar += primitive_rule_name(parameter.type) + "\n";
                continue;
            }

            grammar += enum_rule_name(tool_index, param_index) + "\n";
            grammar += enum_rule_name(tool_index, param_index) + " ::= ";
            for (size_t enum_index = 0; enum_index < parameter.enum_values.size(); ++enum_index) {
                if (enum_index > 0) {
                    grammar += " | ";
                }
                grammar += enum_literal(parameter.enum_values[enum_index], parameter.type);
            }
            grammar += "\n";
        }
    }

    static void append_optional_rules(std::string& grammar, size_t tool_index,
                                      const std::vector<ToolParameter>& parameters,
                                      size_t start_index) {
        for (size_t index = start_index; index <= parameters.size(); ++index) {
            if (index == parameters.size()) {
                grammar += start_rule_name(tool_index, index) + " ::= ws\n";
                grammar += cont_rule_name(tool_index, index) + " ::= ws\n";
                continue;
            }

            grammar += start_rule_name(tool_index, index) + " ::= ws | " +
                       param_rule_name(tool_index, index) + " " + cont_rule_name(tool_index, index + 1) +
                       " | " + start_rule_name(tool_index, index + 1) + "\n";
            grammar += cont_rule_name(tool_index, index) + " ::= ws | ws " + literal(",") +
                       " ws " + param_rule_name(tool_index, index) + " " +
                       cont_rule_name(tool_index, index + 1) + " | " +
                       cont_rule_name(tool_index, index + 1) + "\n";
        }
    }

    static size_t count_required_prefix(const std::vector<ToolParameter>& parameters) {
        size_t count = 0;
        while (count < parameters.size() && parameters[count].required) {
            ++count;
        }
        return count;
    }

    static std::string build_required_sequence(size_t tool_index, size_t begin, size_t end) {
        std::string sequence;
        for (size_t index = begin; index < end; ++index) {
            if (!sequence.empty()) {
                sequence += " ws " + literal(",") + " ws ";
            }
            sequence += param_rule_name(tool_index, index);
        }
        return sequence;
    }

    static std::string primitive_rule_name(ToolValueType type) {
        switch (type) {
        case ToolValueType::Integer:
            return "integer";
        case ToolValueType::Number:
            return "number";
        case ToolValueType::String:
            return "string";
        case ToolValueType::Boolean:
            return "boolean";
        }
        return "string";
    }

    static std::string enum_literal(const nlohmann::json& value, ToolValueType type) {
        if (type == ToolValueType::String) {
            return json_string_literal(value.get<std::string>());
        }
        return literal(value.dump());
    }

    static std::string tool_rule_name(size_t tool_index) {
        return "tool-" + std::to_string(tool_index);
    }

    static std::string args_rule_name(size_t tool_index) {
        return "tool-" + std::to_string(tool_index) + "-args";
    }

    static std::string param_rule_name(size_t tool_index, size_t param_index) {
        return "tool-" + std::to_string(tool_index) + "-param-" + std::to_string(param_index);
    }

    static std::string enum_rule_name(size_t tool_index, size_t param_index) {
        return "tool-" + std::to_string(tool_index) + "-enum-" + std::to_string(param_index);
    }

    static std::string start_rule_name(size_t tool_index, size_t param_index) {
        return "tool-" + std::to_string(tool_index) + "-start-" + std::to_string(param_index);
    }

    static std::string cont_rule_name(size_t tool_index, size_t param_index) {
        return "tool-" + std::to_string(tool_index) + "-cont-" + std::to_string(param_index);
    }

    static std::string json_string_literal(const std::string& value) {
        return literal(nlohmann::json(value).dump());
    }

    static std::string literal(const std::string& value) {
        std::string escaped;
        escaped.reserve(value.size());
        for (char c : value) {
            if (c == '"' || c == '\\') {
                escaped.push_back('\\');
            }
            escaped.push_back(c);
        }
        return "\"" + escaped + "\"";
    }

    static std::string primitive_rules() {
        return "integer ::= \"-\"? [0-9]+\n"
               "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n"
               "string ::= \"\\\"\" [^\"\\\\]* (\"\\\\\" [^\\x00] [^\"\\\\]*)* \"\\\"\"\n"
               "boolean ::= \"true\" | \"false\"\n"
               "ws ::= [ \\t\\n]*\n";
    }
};

} // namespace zoo::tools
