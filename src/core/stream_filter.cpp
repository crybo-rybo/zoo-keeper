#include "zoo/internal/core/stream_filter.hpp"
#include <common.h>

namespace zoo::core {

ToolCallTriggerMatcher::ToolCallTriggerMatcher(
    const std::vector<common_grammar_trigger>& triggers) {
    word_triggers_.reserve(triggers.size());
    regex_triggers_.reserve(triggers.size());

    for (const auto& trigger : triggers) {
        if (trigger.value.empty()) {
            continue;
        }

        switch (trigger.type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            word_triggers_.push_back(trigger.value);
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            try {
                regex_triggers_.push_back(
                    RegexTrigger{std::regex(trigger.value),
                                 trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL});
            } catch (const std::regex_error&) {
                // Malformed pattern: ignore it so runtime generation can continue.
            }
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            break;
        }
    }
}

bool ToolCallTriggerMatcher::is_detected(std::string_view text) const {
    for (const auto& trigger : word_triggers_) {
        if (text.find(trigger) != std::string_view::npos) {
            return true;
        }
    }

    for (const auto& trigger : regex_triggers_) {
        if (trigger.full_match) {
            if (std::regex_match(text.begin(), text.end(), trigger.regex)) {
                return true;
            }
            continue;
        }

        if (std::regex_search(text.begin(), text.end(), trigger.regex)) {
            return true;
        }
    }

    return false;
}

bool is_tool_trigger_detected(const std::string& text,
                              const std::vector<common_grammar_trigger>& triggers) {
    return ToolCallTriggerMatcher(triggers).is_detected(text);
}

std::vector<std::string>
extract_word_triggers(const std::vector<common_grammar_trigger>& triggers) {
    return ToolCallTriggerMatcher(triggers).word_triggers();
}

size_t ToolCallWordTriggerFilter::first_trigger_match(std::string_view text) const {
    size_t earliest = std::string::npos;
    for (const auto& trigger : borrowed_triggers_) {
        const size_t pos = text.find(trigger);
        if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
            earliest = pos;
        }
    }
    return earliest;
}

size_t ToolCallWordTriggerFilter::buffered_suffix_len(std::string_view text) const {
    size_t longest = 0;
    for (const auto& trigger : borrowed_triggers_) {
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
