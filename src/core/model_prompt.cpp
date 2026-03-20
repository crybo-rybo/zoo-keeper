/**
 * @file model_prompt.cpp
 * @brief Incremental prompt rendering and KV-cache bookkeeping for `Model`.
 */

#include "zoo/core/model.hpp"
#include "zoo/core/model_tool_calling_state.hpp"

#include "zoo/internal/core/prompt_bookkeeping.hpp"

#include <chat.h>
#include <llama.h>

namespace zoo::core {

namespace {

/// Converts zoo::Message history to common_chat_msg for the common layer.
std::vector<common_chat_msg> to_chat_msgs(const std::vector<Message>& messages) {
    std::vector<common_chat_msg> result;
    result.reserve(messages.size());
    for (const auto& msg : messages) {
        common_chat_msg cm;
        cm.role = role_to_string(msg.role);
        cm.content = msg.content;

        if (msg.tool_call_id.has_value()) {
            cm.tool_call_id = *msg.tool_call_id;
        }

        for (const auto& tc : msg.tool_calls) {
            cm.tool_calls.push_back({tc.name, tc.arguments_json, tc.id});
        }

        result.push_back(std::move(cm));
    }
    return result;
}

} // namespace

Expected<std::string> Model::render_prompt_delta() {
    auto chat_msgs = to_chat_msgs(messages_);

    // Build inputs for the template system.
    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_msgs);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    // If tools are registered, include them so the template can format them.
    if (tool_state_) {
        inputs.tools = tool_state_->tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    common_chat_params params;
    try {
        params = common_chat_templates_apply(chat_templates_.get(), inputs);
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed,
                  std::string("common_chat_templates_apply failed: ") + e.what()});
    }

    const std::string& new_prompt = params.prompt;
    const int new_len = static_cast<int>(new_prompt.size());

    if (new_len == 0) {
        return std::string{};
    }

    if (rendered_prompt_requires_kv_reset(prompt_state_.committed_prompt_len, new_len)) {
        clear_kv_cache();
    }

    // Extract the delta since the last committed prompt position.
    std::string delta;
    if (prompt_state_.committed_prompt_len < new_len) {
        delta = new_prompt.substr(static_cast<size_t>(prompt_state_.committed_prompt_len));
    } else if (prompt_state_.committed_prompt_len == 0) {
        delta = new_prompt;
    }

    // Cache the full rendered prompt for finalization.
    prompt_state_.rendered_prompt = new_prompt;
    prompt_state_.dirty = false;

    // If we have tool state, fully refresh the current format/parsing/grammar
    // state from this render pass. The template output can vary with history.
    if (tool_state_) {
        common_peg_arena parser;
        try {
            if (!params.parser.empty()) {
                parser.load(params.parser);
            }
        } catch (const std::exception& e) {
            return std::unexpected(
                Error{ErrorCode::TemplateRenderFailed,
                      std::string("Failed to deserialize tool parser: ") + e.what()});
        }

        tool_state_->format = params.format;
        tool_state_->grammar = params.grammar;
        tool_state_->grammar_lazy = params.grammar_lazy;
        tool_state_->grammar_triggers = std::move(params.grammar_triggers);
        tool_state_->preserved_tokens = std::move(params.preserved_tokens);
        tool_state_->additional_stops = std::move(params.additional_stops);
        tool_state_->thinking_forced_open = params.thinking_forced_open;
        tool_state_->parser = std::move(parser);
        tool_grammar_str_ = tool_state_->grammar;
    }

    return delta;
}

void Model::finalize_response() {
    auto chat_msgs = to_chat_msgs(messages_);

    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_msgs);
    inputs.add_generation_prompt = false;
    inputs.use_jinja = true;

    if (tool_state_) {
        inputs.tools = tool_state_->tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    common_chat_params params;
    try {
        params = common_chat_templates_apply(chat_templates_.get(), inputs);
    } catch (const std::exception&) {
        return;
    }

    const int new_prev_len = static_cast<int>(params.prompt.size());
    commit_rendered_prompt(prompt_state_.committed_prompt_len, new_prev_len);
}

void Model::clear_kv_cache() {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_.get()), false);
    }
    prompt_state_.committed_prompt_len = 0;
}

void Model::note_history_append() noexcept {
    note_history_mutation(PromptHistoryMutation::Append, prompt_state_.dirty,
                          prompt_state_.committed_prompt_len);
}

void Model::note_history_rewrite() noexcept {
    note_history_mutation(PromptHistoryMutation::Rewrite, prompt_state_.dirty,
                          prompt_state_.committed_prompt_len);
    clear_kv_cache();
}

void Model::note_history_reset() noexcept {
    note_history_mutation(PromptHistoryMutation::Reset, prompt_state_.dirty,
                          prompt_state_.committed_prompt_len);
    clear_kv_cache();
}

} // namespace zoo::core
