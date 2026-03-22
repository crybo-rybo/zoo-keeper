/**
 * @file model_tool_calling_state.hpp
 * @brief Private definition of Model::ToolCallingState for implementation files.
 *
 * This header is intentionally NOT installed — it is only used by model*.cpp
 * files that need the complete type for pimpl'd tool calling state.
 */

#pragma once

#include "model.hpp"
#include "zoo/internal/core/stream_filter.hpp"

#include <chat.h>
#include <common.h>
#include <string>
#include <vector>

namespace zoo::core {

struct Model::ToolCallingState {
    std::vector<common_chat_tool> tools;
    common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    ToolCallTriggerMatcher trigger_matcher;
    std::vector<std::string> preserved_tokens;
    std::vector<std::string> additional_stops;
    bool thinking_forced_open = false;
    common_peg_arena parser;
};

} // namespace zoo::core
