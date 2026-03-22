/**
 * @file stream_filter.hpp
 * @brief Helper for detecting tool-call grammar triggers in streamed text.
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

// Forward declare to avoid pulling in llama.cpp headers.
struct common_grammar_trigger;

namespace zoo::core {

/// Returns true if any non-word grammar trigger is matched in the accumulated text.
///
/// Supports PATTERN triggers (regex search) and PATTERN_FULL triggers (regex
/// match against the full text). WORD triggers are handled by
/// `ToolCallWordTriggerFilter` so split prefixes can be buffered safely.
/// TOKEN triggers are skipped as they operate at the token-id level.
bool is_tool_trigger_detected(const std::string& text,
                              const std::vector<common_grammar_trigger>& triggers);

/// Extracts literal WORD triggers from the grammar trigger list.
std::vector<std::string> extract_word_triggers(const std::vector<common_grammar_trigger>& triggers);

/// Streams visible text while buffering possible literal trigger prefixes.
class ToolCallWordTriggerFilter {
  public:
    explicit ToolCallWordTriggerFilter(std::vector<std::string> word_triggers)
        : word_triggers_(std::move(word_triggers)) {}

    std::string consume(std::string_view token);
    std::string finalize();

    bool suppressing() const noexcept {
        return suppressing_;
    }

  private:
    size_t buffered_suffix_len(std::string_view text) const;
    size_t first_trigger_match(std::string_view text) const;

    std::vector<std::string> word_triggers_;
    std::string pending_;
    bool suppressing_ = false;
};

} // namespace zoo::core
