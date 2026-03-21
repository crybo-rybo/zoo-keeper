#include "zoo/internal/core/stream_filter.hpp"
#include <common.h>
#include <regex>

namespace zoo::core {

bool is_tool_trigger_detected(const std::string& text,
                              const std::vector<common_grammar_trigger>& triggers) {
    for (const auto& trigger : triggers) {
        if (trigger.value.empty()) {
            continue;
        }
        switch (trigger.type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            if (text.find(trigger.value) != std::string::npos) {
                return true;
            }
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            try {
                if (std::regex_search(text, std::regex(trigger.value))) {
                    return true;
                }
            } catch (const std::regex_error&) {
                // Malformed pattern — skip.
            }
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            try {
                if (std::regex_match(text, std::regex(trigger.value))) {
                    return true;
                }
            } catch (const std::regex_error&) {
                // Malformed pattern — skip.
            }
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            // Token-id triggers cannot be checked at the text level.
            break;
        }
    }
    return false;
}

} // namespace zoo::core
