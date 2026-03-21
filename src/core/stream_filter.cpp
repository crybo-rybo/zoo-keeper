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

std::vector<std::string>
extract_word_triggers(const std::vector<common_grammar_trigger>& triggers) {
    std::vector<std::string> words;
    for (const auto& trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD && !trigger.value.empty()) {
            words.push_back(trigger.value);
        }
    }
    return words;
}

size_t ToolCallWordTriggerFilter::first_trigger_match(std::string_view text) const {
    size_t earliest = std::string::npos;
    for (const auto& trigger : word_triggers_) {
        const size_t pos = text.find(trigger);
        if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
            earliest = pos;
        }
    }
    return earliest;
}

size_t ToolCallWordTriggerFilter::buffered_suffix_len(std::string_view text) const {
    size_t longest = 0;
    for (const auto& trigger : word_triggers_) {
        const size_t max_check = std::min(text.size(), trigger.size() - 1);
        for (size_t len = 1; len <= max_check; ++len) {
            if (text.substr(text.size() - len) == std::string_view(trigger).substr(0, len)) {
                longest = std::max(longest, len);
            }
        }
    }
    return longest;
}

std::string ToolCallWordTriggerFilter::consume(std::string_view token) {
    if (suppressing_) {
        return {};
    }

    std::string candidate = pending_;
    candidate.append(token);

    const size_t trigger_pos = first_trigger_match(candidate);
    if (trigger_pos != std::string::npos) {
        suppressing_ = true;
        pending_.clear();
        return candidate.substr(0, trigger_pos);
    }

    const size_t suffix_len = buffered_suffix_len(candidate);
    pending_ = candidate.substr(candidate.size() - suffix_len);
    candidate.resize(candidate.size() - suffix_len);
    return candidate;
}

std::string ToolCallWordTriggerFilter::finalize() {
    if (suppressing_) {
        return {};
    }
    std::string trailing = std::move(pending_);
    pending_.clear();
    return trailing;
}

} // namespace zoo::core
