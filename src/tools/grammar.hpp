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
 * @brief Builds a GBNF grammar that constrains model output to a JSON schema shape.
 *
 * Used by the structured-extraction path to generate an immediately-active
 * grammar for schema-constrained generation. Tool calling grammars are handled
 * by the llama.cpp common layer via `Model::set_tool_calling()`.
 */
class GrammarBuilder {
  public:
    /**
     * @brief Builds a grammar for a standalone JSON schema.
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

        grammar += "root ::= " + literal("{") + " ws " + prefix + "-args ws " + literal("}") + "\n";

        append_prefixed_parameter_rules(grammar, prefix, parameters);

        const size_t required_count = count_required_prefix(parameters);

        grammar += prefix + "-args ::= ";
        if (required_count == parameters.size()) {
            grammar += build_prefixed_required_sequence(prefix, 0, required_count);
        } else if (required_count == 0) {
            grammar += prefix + "-start-0";
        } else {
            grammar += build_prefixed_required_sequence(prefix, 0, required_count) + " " + prefix +
                       "-cont-" + std::to_string(required_count);
        }
        grammar += "\n";

        if (required_count < parameters.size()) {
            append_prefixed_optional_rules(grammar, prefix, parameters, required_count);
        }

        grammar += primitive_rules();
        return grammar;
    }

  private:
    static void append_prefixed_parameter_rules(std::string& grammar, const std::string& prefix,
                                                const std::vector<ToolParameter>& parameters) {
        for (size_t param_index = 0; param_index < parameters.size(); ++param_index) {
            const auto& parameter = parameters[param_index];
            const std::string p_rule = prefix + "-param-" + std::to_string(param_index);
            grammar += p_rule + " ::= " + json_string_literal(parameter.name) + " ws " +
                       literal(":") + " ws ";

            if (parameter.enum_values.empty()) {
                grammar += tool_value_type_name(parameter.type) + std::string("\n");
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

    static void append_prefixed_optional_rules(std::string& grammar, const std::string& prefix,
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

            grammar +=
                start_r + " ::= ws | " + param_r + " " + next_cont + " | " + next_start + "\n";
            grammar += cont_r + " ::= ws | ws " + literal(",") + " ws " + param_r + " " +
                       next_cont + " | " + next_cont + "\n";
        }
    }

    static size_t count_required_prefix(const std::vector<ToolParameter>& parameters) {
        size_t count = 0;
        while (count < parameters.size() && parameters[count].required) {
            ++count;
        }
        return count;
    }

    static std::string build_prefixed_required_sequence(const std::string& prefix, size_t begin,
                                                        size_t end) {
        std::string sequence;
        for (size_t index = begin; index < end; ++index) {
            if (!sequence.empty()) {
                sequence += " ws " + literal(",") + " ws ";
            }
            sequence += prefix + "-param-" + std::to_string(index);
        }
        return sequence;
    }

    static std::string enum_literal(const nlohmann::json& value, ToolValueType type) {
        if (type == ToolValueType::String) {
            return json_string_literal(value.get<std::string>());
        }
        return literal(value.dump());
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
