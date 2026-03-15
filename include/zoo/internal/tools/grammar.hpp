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

    /**
     * @brief Builds a grammar for a standalone JSON schema (no tool-call sentinels).
     *
     * @param parameters Normalized parameter vector describing the output schema.
     * @return A GBNF grammar rooted at `root`, or an empty string when no
     *         parameters are provided.
     */
    static std::string build_schema(const std::vector<ToolParameter>& parameters) {
        if (parameters.empty()) {
            std::string grammar;
            grammar += "root ::= " + literal("{") + " ws " + literal("}") + "\n";
            grammar += primitive_rules();
            return grammar;
        }

        std::string grammar;
        const std::string prefix = "schema-0";

        grammar += "root ::= " + literal("{") + " ws " + prefix + "-args ws " +
                   literal("}") + "\n";

        append_prefixed_parameter_rules(grammar, prefix, parameters);

        const size_t required_count = count_required_prefix(parameters);

        grammar += prefix + "-args ::= ";
        if (required_count == parameters.size()) {
            grammar += build_prefixed_required_sequence(prefix, 0, required_count);
        } else if (required_count == 0) {
            grammar += prefix + "-start-0";
        } else {
            grammar += build_prefixed_required_sequence(prefix, 0, required_count) + " " +
                       prefix + "-cont-" + std::to_string(required_count);
        }
        grammar += "\n";

        if (required_count < parameters.size()) {
            append_prefixed_optional_rules(grammar, prefix, parameters, required_count);
        }

        grammar += primitive_rules();
        return grammar;
    }

  private:
    static void append_tool_rules(std::string& grammar, size_t tool_index,
                                  const ToolMetadata& tool) {
        const std::string prefix = "tool-" + std::to_string(tool_index);

        grammar += tool_rule_name(tool_index) + " ::= " + literal("{") + " ws " +
                   json_string_literal("name") + " ws " + literal(":") + " ws " +
                   json_string_literal(tool.name) + " ws " + literal(",") + " ws " +
                   json_string_literal("arguments") + " ws " + literal(":") + " ws " +
                   literal("{") + " ws ";

        if (!tool.parameters.empty()) {
            grammar += prefix + "-args ws ";
        }

        grammar += literal("}") + " ws " + literal("}") + "\n";

        if (tool.parameters.empty()) {
            return;
        }

        const size_t required_count = count_required_prefix(tool.parameters);
        append_prefixed_parameter_rules(grammar, prefix, tool.parameters);

        grammar += prefix + "-args ::= ";
        if (required_count == tool.parameters.size()) {
            grammar += build_prefixed_required_sequence(prefix, 0, required_count);
        } else if (required_count == 0) {
            grammar += prefix + "-start-0";
        } else {
            grammar += build_prefixed_required_sequence(prefix, 0, required_count) + " " +
                       prefix + "-cont-" + std::to_string(required_count);
        }
        grammar += "\n";

        if (required_count < tool.parameters.size()) {
            append_prefixed_optional_rules(grammar, prefix, tool.parameters, required_count);
        }
    }

    static void append_prefixed_parameter_rules(std::string& grammar,
                                                const std::string& prefix,
                                                const std::vector<ToolParameter>& parameters) {
        for (size_t param_index = 0; param_index < parameters.size(); ++param_index) {
            const auto& parameter = parameters[param_index];
            const std::string p_rule = prefix + "-param-" + std::to_string(param_index);
            grammar += p_rule + " ::= " + json_string_literal(parameter.name) + " ws " +
                       literal(":") + " ws ";

            if (parameter.enum_values.empty()) {
                grammar += primitive_rule_name(parameter.type) + "\n";
                continue;
            }

            const std::string e_rule = prefix + "-enum-" + std::to_string(param_index);
            grammar += e_rule + "\n";
            grammar += e_rule + " ::= ";
            for (size_t enum_index = 0; enum_index < parameter.enum_values.size(); ++enum_index) {
                if (enum_index > 0) {
                    grammar += " | ";
                }
                grammar += enum_literal(parameter.enum_values[enum_index], parameter.type);
            }
            grammar += "\n";
        }
    }

    static void append_prefixed_optional_rules(std::string& grammar,
                                               const std::string& prefix,
                                               const std::vector<ToolParameter>& parameters,
                                               size_t start_index) {
        for (size_t index = start_index; index <= parameters.size(); ++index) {
            const std::string start_r = prefix + "-start-" + std::to_string(index);
            const std::string cont_r = prefix + "-cont-" + std::to_string(index);
            const std::string param_r = prefix + "-param-" + std::to_string(index);

            if (index == parameters.size()) {
                grammar += start_r + " ::= ws\n";
                grammar += cont_r + " ::= ws\n";
                continue;
            }

            const std::string next_cont = prefix + "-cont-" + std::to_string(index + 1);
            const std::string next_start = prefix + "-start-" + std::to_string(index + 1);

            grammar += start_r + " ::= ws | " + param_r + " " + next_cont + " | " +
                       next_start + "\n";
            grammar += cont_r + " ::= ws | ws " + literal(",") + " ws " +
                       param_r + " " + next_cont + " | " + next_cont + "\n";
        }
    }

    static size_t count_required_prefix(const std::vector<ToolParameter>& parameters) {
        size_t count = 0;
        while (count < parameters.size() && parameters[count].required) {
            ++count;
        }
        return count;
    }

    static std::string build_prefixed_required_sequence(const std::string& prefix,
                                                        size_t begin, size_t end) {
        std::string sequence;
        for (size_t index = begin; index < end; ++index) {
            if (!sequence.empty()) {
                sequence += " ws " + literal(",") + " ws ";
            }
            sequence += prefix + "-param-" + std::to_string(index);
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
