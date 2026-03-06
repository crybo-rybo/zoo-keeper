#pragma once

#include "types.hpp"
#include <nlohmann/json.hpp>
#include <atomic>
#include <string>
#include <optional>

namespace zoo::tools {

class ToolCallParser {
public:
    struct ParseResult {
        std::optional<ToolCall> tool_call;
        std::string text_before;
    };

    static ParseResult parse(const std::string& output) {
        ParseResult result;

        auto pos = output.find('{');
        while (pos != std::string::npos) {
            auto end_pos = find_json_object_end(output, pos);
            if (end_pos != std::string::npos) {
                try {
                    auto begin_it = output.cbegin() + static_cast<std::string::difference_type>(pos);
                    auto end_it = output.cbegin() + static_cast<std::string::difference_type>(end_pos + 1);
                    auto j = nlohmann::json::parse(begin_it, end_it);

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
                }
            }
            pos = output.find('{', pos + 1);
        }

        result.text_before = output;
        return result;
    }

private:
    static size_t find_json_object_end(const std::string& text, size_t start) {
        if (start >= text.size() || text[start] != '{') return std::string::npos;

        int depth = 0;
        bool in_string = false;
        bool escape_next = false;

        for (size_t i = start; i < text.size(); ++i) {
            char c = text[i];

            if (escape_next) { escape_next = false; continue; }
            if (c == '\\' && in_string) { escape_next = true; continue; }
            if (c == '"') { in_string = !in_string; continue; }

            if (!in_string) {
                if (c == '{') ++depth;
                else if (c == '}') {
                    --depth;
                    if (depth == 0) return i;
                }
            }
        }

        return std::string::npos;
    }

    static std::string generate_id() {
        static std::atomic<int> counter{0};
        return "call_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed) + 1);
    }
};

} // namespace zoo::tools
