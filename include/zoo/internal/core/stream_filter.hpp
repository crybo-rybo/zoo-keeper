/**
 * @file stream_filter.hpp
 * @brief Helper for detecting tool-call grammar triggers in streamed text.
 */

#pragma once

#include <string>
#include <vector>

// Forward declare to avoid pulling in llama.cpp headers.
struct common_grammar_trigger;

namespace zoo::core {

/// Returns true if any grammar trigger is matched in the accumulated text.
///
/// Supports WORD triggers (literal substring match), PATTERN triggers (regex
/// search), and PATTERN_FULL triggers (regex match against the full text).
/// TOKEN triggers are skipped as they operate at the token-id level.
bool is_tool_trigger_detected(const std::string& text,
                              const std::vector<common_grammar_trigger>& triggers);

} // namespace zoo::core
