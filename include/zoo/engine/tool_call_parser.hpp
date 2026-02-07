#pragma once

#include "../types.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <optional>

namespace zoo {

// ============================================================================
// ToolCall struct
// ============================================================================

struct ToolCall {
    std::string id;
    std::string name;
    nlohmann::json arguments;

    bool operator==(const ToolCall& other) const {
        return id == other.id && name == other.name && arguments == other.arguments;
    }
    bool operator!=(const ToolCall& other) const { return !(*this == other); }
};

namespace engine {

// ============================================================================
// ToolCallParser
// ============================================================================

class ToolCallParser {
public:
    struct ParseResult {
        std::optional<ToolCall> tool_call;
        std::string text_before;  // Text before the tool call JSON
    };

    /**
     * Parse model output to detect a tool call.
     *
     * Looks for a JSON object with "name" and "arguments" fields.
     * Returns the parsed ToolCall if found, along with any text before it.
     */
    static ParseResult parse(const std::string& output) {
        ParseResult result;

        // Find the first '{' that could be a tool call JSON
        auto pos = output.find('{');
        while (pos != std::string::npos) {
            // Try to parse JSON starting at this position
            auto json_str = extract_json_object(output, pos);
            if (!json_str.empty()) {
                try {
                    auto j = nlohmann::json::parse(json_str);

                    // Check if it has the tool call structure
                    if (j.is_object() && j.contains("name") && j.contains("arguments")) {
                        ToolCall tc;
                        tc.name = j["name"].get<std::string>();
                        tc.arguments = j["arguments"];
                        tc.id = j.value("id", generate_id());

                        result.tool_call = std::move(tc);
                        result.text_before = output.substr(0, pos);
                        return result;
                    }
                } catch (const nlohmann::json::exception&) {
                    // Not valid JSON, try next '{'
                }
            }

            pos = output.find('{', pos + 1);
        }

        // No tool call found
        result.text_before = output;
        return result;
    }

private:
    /**
     * Extract a balanced JSON object starting at pos.
     * Returns empty string if braces are unbalanced.
     */
    static std::string extract_json_object(const std::string& text, size_t start) {
        if (start >= text.size() || text[start] != '{') return "";

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
                if (c == '{') ++depth;
                else if (c == '}') {
                    --depth;
                    if (depth == 0) {
                        return text.substr(start, i - start + 1);
                    }
                }
            }
        }

        return "";  // Unbalanced
    }

    static std::string generate_id() {
        static int counter = 0;
        return "call_" + std::to_string(++counter);
    }
};

} // namespace engine
} // namespace zoo
