/**
 * @file model_prompt.cpp
 * @brief Incremental prompt rendering and KV-cache bookkeeping for `Model`.
 */

#include "core/model_impl.hpp"
#include "zoo/core/model.hpp"

#include "core/prompt_bookkeeping.hpp"

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

        if (!msg.tool_call_id.empty()) {
            cm.tool_call_id = msg.tool_call_id;
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
    auto chat_msgs = to_chat_msgs(impl_->messages_);

    // Build inputs for the template system.
    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_msgs);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    // If tools are registered, include them so the template can format them.
    if (impl_->tool_state_) {
        inputs.tools = impl_->tool_state_->tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    // Thinking is disabled globally until zoo surfaces a first-class API for
    // it. See docs/adr/007-thinking-disabled-by-default.md.
    inputs.enable_thinking = false;

    common_chat_params params;
    try {
        params = common_chat_templates_apply(impl_->chat_templates_.get(), inputs);
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

    if (rendered_prompt_requires_kv_reset(impl_->prompt_state_.committed_prompt_len, new_len)) {
        clear_kv_cache();
    }

    // Extract the delta since the last committed prompt position.
    std::string delta;
    if (impl_->prompt_state_.committed_prompt_len < new_len) {
        delta = new_prompt.substr(static_cast<size_t>(impl_->prompt_state_.committed_prompt_len));
    } else if (impl_->prompt_state_.committed_prompt_len == 0) {
        delta = new_prompt;
    }

    // Cache the full rendered prompt for finalization.
    impl_->prompt_state_.rendered_prompt = new_prompt;
    impl_->prompt_state_.dirty = false;

    // If native tool calling is active, fully refresh the current
    // format/parsing/grammar state from this render pass. The template output
    // can vary with history. Skip this when in Schema mode (extraction) to
    // avoid overwriting the caller's schema grammar with tool-call grammar.
    if (impl_->grammar_mode_ == Impl::GrammarMode::NativeToolCall && impl_->tool_state_) {
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

        impl_->tool_state_->format = params.format;
        impl_->tool_state_->grammar = params.grammar;
        impl_->tool_state_->grammar_lazy = params.grammar_lazy;
        impl_->tool_state_->grammar_triggers = std::move(params.grammar_triggers);
        impl_->tool_state_->trigger_matcher =
            ToolCallTriggerMatcher(impl_->tool_state_->grammar_triggers);
        impl_->tool_state_->preserved_tokens = std::move(params.preserved_tokens);
        impl_->tool_state_->additional_stops = std::move(params.additional_stops);
        impl_->tool_state_->thinking_forced_open = params.thinking_forced_open;
        impl_->tool_state_->parser = std::move(parser);
        impl_->tool_grammar_str_ = impl_->tool_state_->grammar;
    }

    return delta;
}

void Model::finalize_response() {
    auto chat_msgs = to_chat_msgs(impl_->messages_);

    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_msgs);
    inputs.add_generation_prompt = false;
    inputs.use_jinja = true;

    if (impl_->tool_state_) {
        inputs.tools = impl_->tool_state_->tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    // See docs/adr/007-thinking-disabled-by-default.md.
    inputs.enable_thinking = false;

    common_chat_params params;
    try {
        params = common_chat_templates_apply(impl_->chat_templates_.get(), inputs);
    } catch (const std::exception&) {
        return;
    }

    const int new_prev_len = static_cast<int>(params.prompt.size());
    commit_rendered_prompt(impl_->prompt_state_.committed_prompt_len, new_prev_len);
}

void Model::clear_kv_cache() {
    if (impl_->ctx_) {
        llama_memory_clear(llama_get_memory(impl_->ctx_.get()), false);
    }
    impl_->prompt_state_.committed_prompt_len = 0;
}

void Model::note_history_append() noexcept {
    note_history_mutation(PromptHistoryMutation::Append, impl_->prompt_state_.dirty,
                          impl_->prompt_state_.committed_prompt_len);
}

void Model::note_history_rewrite() noexcept {
    note_history_mutation(PromptHistoryMutation::Rewrite, impl_->prompt_state_.dirty,
                          impl_->prompt_state_.committed_prompt_len);
    clear_kv_cache();
}

void Model::note_history_reset() noexcept {
    note_history_mutation(PromptHistoryMutation::Reset, impl_->prompt_state_.dirty,
                          impl_->prompt_state_.committed_prompt_len);
    clear_kv_cache();
}

} // namespace zoo::core
