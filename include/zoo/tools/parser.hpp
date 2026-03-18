/**
 * @file parser.hpp
 * @brief Parsers that recover tool calls from unconstrained or sentinel-delimited output.
 */

#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>

namespace zoo::tools {

/**
 * @brief Extracts tool calls from model output.
 *
 * The parser supports both heuristic brace-based extraction and explicit
 * `<tool_call>...</tool_call>` sentinel tags used by grammar-constrained mode.
 */
class ToolCallParser {
  public:
    /**
     * @brief Result of attempting to parse a tool call from model output.
     */
    struct ParseResult {
        std::optional<ToolCall> tool_call; ///< Parsed tool call when one is detected.
        std::string text_before; ///< Text that appears before the parsed tool call, or the original
                                 ///< output on failure.
    };

    /**
     * @brief Extracts the first JSON object shaped like a tool call.
     *
     * @param output Raw model output that may contain visible text and JSON.
     * @return Parsed tool call plus the text that precedes it.
     */
    static ParseResult parse(const std::string& output) {
        ParseResult result;

        auto pos = output.find('{');
        while (pos != std::string::npos) {
            auto end_pos = find_json_object_end(output, pos);
            if (end_pos != std::string::npos) {
                try {
                    auto begin_it =
                        output.cbegin() + static_cast<std::string::difference_type>(pos);
                    auto end_it =
                        output.cbegin() + static_cast<std::string::difference_type>(end_pos + 1);
                    auto j = nlohmann::json::parse(begin_it, end_it);

                    if (j.is_object() && j.contains("name") && j.contains("arguments")) {
                        ToolCall tc;
                        tc.name = j["name"].get<std::string>();
                        tc.arguments = std::move(j["arguments"]);
                        tc.id = j.value("id", generate_id(std::string_view(output).substr(
                                                  pos, end_pos - pos + 1)));

                        result.tool_call = std::move(tc);
                        result.text_before = output.substr(0, pos);
                        return result;
                    }
                } catch (const nlohmann::json::exception&) {
                }
            }
            pos = output.find('{', pos + 1);
        }

        result.text_before = output;
        return result;
    }

  private:
    /// Finds the matching closing brace for the JSON object that starts at `start`.
    static size_t find_json_object_end(const std::string& text, size_t start) {
        if (start >= text.size() || text[start] != '{')
            return std::string::npos;

        int depth = 0;
        bool in_string = false;
        bool escape_next = false;

        for (size_t i = start; i < text.size(); ++i) {
            char c = text[i];

            if (escape_next) {
                escape_next = false;
                continue;
            }
            if (c == '\\' && in_string) {
                escape_next = true;
                continue;
            }
            if (c == '"') {
                in_string = !in_string;
                continue;
            }

            if (!in_string) {
                if (c == '{')
                    ++depth;
                else if (c == '}') {
                    --depth;
                    if (depth == 0)
                        return i;
                }
            }
        }

        return std::string::npos;
    }

    /// Generates a stable fallback tool-call identifier when the model omits one.
    static std::string generate_id(std::string_view text) {
        constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
        constexpr uint64_t kPrime = 1099511628211ull;

        uint64_t hash = kOffsetBasis;
        for (unsigned char c : text) {
            hash ^= c;
            hash *= kPrime;
        }
        return "call_" + std::to_string(hash);
    }
};

} // namespace zoo::tools
