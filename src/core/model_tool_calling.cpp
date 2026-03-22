/**
 * @file model_tool_calling.cpp
 * @brief Template-driven tool calling: prepare, parse, and grammar management.
 */

#include "zoo/core/model.hpp"
#include "zoo/core/model_tool_calling_state.hpp"
#include "zoo/internal/log.hpp"

#include <chat.h>
#include <common.h>
#include <llama.h>

namespace zoo::core {

// ---------------------------------------------------------------------------
// set_tool_calling
// ---------------------------------------------------------------------------

bool Model::set_tool_calling(const std::vector<CoreToolInfo>& tools) {
    if (tools.empty()) {
        clear_tool_grammar();
        return true;
    }

    // Convert CoreToolInfo → common_chat_tool
    std::vector<common_chat_tool> chat_tools;
    chat_tools.reserve(tools.size());
    for (const auto& t : tools) {
        chat_tools.push_back({t.name, t.description, t.parameters_json});
    }

    // Build a minimal message set to probe the template for grammar/triggers.
    // We need at least one user message for the template to produce output.
    common_chat_templates_inputs inputs;
    inputs.messages = {{
        .role = "user",
        .content = "hello",
    }};
    inputs.tools = chat_tools;
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    common_chat_params params;
    try {
        params = common_chat_templates_apply(chat_templates_.get(), inputs);
    } catch (const std::exception&) {
        ZOO_LOG("info", "native tool calling not available for this model");
        return false;
    }

    // Only support structured native formats — reject generic wrapper and
    // content-only (no tool calling) so there is a single runtime path.
    if (params.format == COMMON_CHAT_FORMAT_CONTENT_ONLY ||
        params.format == COMMON_CHAT_FORMAT_GENERIC) {
        ZOO_LOG("info", "native tool calling not available for this model (format: '%s')",
                common_chat_format_name(params.format));
        return false;
    }

    common_peg_arena parser;
    try {
        if (!params.parser.empty()) {
            parser.load(params.parser);
        }
    } catch (const std::exception&) {
        return false;
    }

    auto state = std::make_unique<ToolCallingState>();
    state->tools = std::move(chat_tools);
    state->format = params.format;
    state->grammar = std::move(params.grammar);
    state->grammar_lazy = params.grammar_lazy;
    state->grammar_triggers = std::move(params.grammar_triggers);
    state->preserved_tokens = std::move(params.preserved_tokens);
    state->additional_stops = std::move(params.additional_stops);
    state->thinking_forced_open = params.thinking_forced_open;
    state->parser = std::move(parser);

    tool_grammar_str_ = state->grammar;
    tool_state_ = std::move(state);

    if (!tool_grammar_str_.empty()) {
        if (!rebuild_sampler_with_tool_grammar()) {
            tool_state_.reset();
            tool_grammar_str_.clear();
            return false;
        }
    }

    grammar_mode_ = GrammarMode::NativeToolCall;
    ZOO_LOG("info", "native tool calling enabled: format '%s' (%zu tools registered)",
            common_chat_format_name(tool_state_->format), tool_state_->tools.size());
    return true;
}

// ---------------------------------------------------------------------------
// parse_tool_response
// ---------------------------------------------------------------------------

Model::ParsedResponse Model::parse_tool_response(const std::string& text) const {
    ParsedResponse result;

    if (!tool_state_) {
        result.content = text;
        return result;
    }

    common_chat_syntax syntax;
    syntax.format = tool_state_->format;
    syntax.thinking_forced_open = tool_state_->thinking_forced_open;
    syntax.parse_tool_calls = true;
    syntax.parser = tool_state_->parser;

    common_chat_msg parsed;
    try {
        parsed = common_chat_parse(text, false, syntax);
    } catch (const std::exception&) {
        result.content = text;
        return result;
    }

    result.content = std::move(parsed.content);
    result.tool_calls.reserve(parsed.tool_calls.size());
    for (auto& tc : parsed.tool_calls) {
        result.tool_calls.push_back(ToolCallInfo{
            std::move(tc.id),
            std::move(tc.name),
            std::move(tc.arguments),
        });
    }

    return result;
}

// ---------------------------------------------------------------------------
// clear_tool_grammar
// ---------------------------------------------------------------------------

void Model::clear_tool_grammar() noexcept {
    if (grammar_mode_ == GrammarMode::None) {
        return;
    }

    tool_grammar_str_.clear();
    tool_state_.reset();
    grammar_mode_ = GrammarMode::None;
    sampler_ = create_sampler_chain();
}

// ---------------------------------------------------------------------------
// tool_calling_format_name
// ---------------------------------------------------------------------------

const char* Model::tool_calling_format_name() const noexcept {
    if (!tool_state_) {
        return "none";
    }
    return common_chat_format_name(tool_state_->format);
}

} // namespace zoo::core
