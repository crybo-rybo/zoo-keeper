#pragma once

#include "../types.hpp"
#include <nlohmann/json.hpp>
#include <atomic>
#include <string>
#include <optional>

namespace zoo {

// ============================================================================
// ToolCall struct
// ============================================================================

/** @brief Represents a parsed tool call extracted from model output. */
struct ToolCall {
    std::string id;              ///< Unique identifier for this tool call invocation
    std::string name;            ///< Name of the tool to invoke
    nlohmann::json arguments;    ///< JSON object of arguments to pass to the tool

    bool operator==(const ToolCall& other) const {
        return id == other.id && name == other.name && arguments == other.arguments;
    }
    bool operator!=(const ToolCall& other) const { return !(*this == other); }
};

namespace engine {

// ============================================================================
// ToolCallParser
// ============================================================================

/**
 * @brief Detects and extracts tool calls from raw model output text.
 *
 * Scans for JSON objects containing "name" and "arguments" fields, which
 * indicate the model is requesting a tool invocation.
 */
class ToolCallParser {
public:
    /** @brief Result of parsing model output for a tool call. */
    struct ParseResult {
        std::optional<ToolCall> tool_call;  ///< The extracted tool call, if any
        std::string text_before;            ///< Text before the tool call JSON
    };

    /**
     * @brief Parse model output to detect a tool call.
     *
     * Looks for a JSON object with "name" and "arguments" fields.
     * Returns the parsed ToolCall if found, along with any text before it.
     *
     * @param output Raw text output from the model
     * @return ParseResult containing the detected tool call and preceding text
     */
    static ParseResult parse(const std::string& output) {
        ParseResult result;

        // Find the first '{' that could be a tool call JSON
        auto pos = output.find('{');
        while (pos != std::string::npos) {
            // Find balanced JSON end position (no substring allocation)
            auto end_pos = find_json_object_end(output, pos);
            if (end_pos != std::string::npos) {
                try {
                    // Parse directly from iterators to avoid intermediate string allocation
                    auto begin_it = output.cbegin() + static_cast<std::string::difference_type>(pos);
                    auto end_it = output.cbegin() + static_cast<std::string::difference_type>(end_pos + 1);
                    auto j = nlohmann::json::parse(begin_it, end_it);

                    // Check if it has the tool call structure
                    if (j.is_object() && j.contains("name") && j.contains("arguments")) {
                        ToolCall tc;
                        tc.name = j["name"].get<std::string>();
                        tc.arguments = std::move(j["arguments"]);
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
     * @brief Find the end position of a balanced JSON object starting at start.
     *
     * Handles string escaping and nested objects to correctly identify the
     * closing brace that matches the opening brace at the start position.
     *
     * @param text The text to search within
     * @param start Starting position (must point to an opening brace '{')
     * @return Position of the matching closing brace, or std::string::npos if unbalanced
     */
    static size_t find_json_object_end(const std::string& text, size_t start) {
        if (start >= text.size() || text[start] != '{') return std::string::npos;

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
                        return i;
                    }
                }
            }
        }

        return std::string::npos;  // Unbalanced
    }

    /**
     * @brief Generate a unique tool call ID.
     *
     * Uses an atomic counter to ensure unique IDs across multiple tool calls.
     * Format: "call_N" where N is a sequential counter.
     *
     * @return Unique tool call identifier
     */
    static std::string generate_id() {
        static std::atomic<int> counter{0};
        return "call_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed) + 1);
    }
};

} // namespace engine
} // namespace zoo
