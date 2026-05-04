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

Expected<std::string> render_prompt_delta(Model::Impl& impl) {
    auto chat_msgs = to_chat_msgs(impl.session_.messages);

    // Build inputs for the template system.
    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_msgs);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;

    // Tool definitions belong to native tool-call generations, not schema extraction overrides.
    if (impl.session_.sampler_policy.is_native_tool_call() && impl.session_.tool_state) {
        inputs.tools = impl.session_.tool_state->tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    // Thinking is disabled globally until zoo surfaces a first-class API for
    // it. See docs/adr/007-thinking-disabled-by-default.md.
    inputs.enable_thinking = false;

    common_chat_params params;
    try {
        params = common_chat_templates_apply(impl.loaded_.chat_templates.get(), inputs);
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

    if (rendered_prompt_requires_kv_reset(impl.session_.prompt_state.committed_prompt_len,
                                          new_len)) {
        clear_kv_cache(impl);
    }

    // Extract the delta since the last committed prompt position.
    std::string delta;
    if (impl.session_.prompt_state.committed_prompt_len < new_len) {
        delta =
            new_prompt.substr(static_cast<size_t>(impl.session_.prompt_state.committed_prompt_len));
    } else if (impl.session_.prompt_state.committed_prompt_len == 0) {
        delta = new_prompt;
    }

    // Cache the full rendered prompt for finalization.
    impl.session_.prompt_state.rendered_prompt = new_prompt;
    impl.session_.prompt_state.dirty = false;

    // If native tool calling is active, fully refresh the current
    // format/parsing/grammar state from this render pass. The template output
    // can vary with history. Skip this when in Schema mode (extraction) to
    // avoid overwriting the caller's schema grammar with tool-call grammar.
    if (impl.session_.sampler_policy.is_native_tool_call() && impl.session_.tool_state) {
        common_chat_parser_params parser_params;
        try {
            parser_params = make_tool_parser_params(params);
        } catch (const std::exception& e) {
            return std::unexpected(
                Error{ErrorCode::TemplateRenderFailed,
                      std::string("Failed to deserialize tool parser: ") + e.what()});
        }

        impl.session_.tool_state->parser_params = std::move(parser_params);
        impl.session_.tool_state->grammar = params.grammar;
        impl.session_.tool_state->grammar_lazy = params.grammar_lazy;
        impl.session_.tool_state->grammar_triggers = std::move(params.grammar_triggers);
        impl.session_.tool_state->trigger_matcher =
            ToolCallTriggerMatcher(impl.session_.tool_state->grammar_triggers);
        impl.session_.tool_state->preserved_tokens = std::move(params.preserved_tokens);
        impl.session_.tool_state->additional_stops = std::move(params.additional_stops);
        impl.session_.sampler_policy =
            Model::Impl::SamplerPolicy::native_tool_call(impl.session_.tool_state->grammar);
    }

    return delta;
}

void Model::finalize_response() {
    auto chat_msgs = to_chat_msgs(impl_->session_.messages);

    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_msgs);
    inputs.add_generation_prompt = false;
    inputs.use_jinja = true;

    if (impl_->session_.sampler_policy.is_native_tool_call() && impl_->session_.tool_state) {
        inputs.tools = impl_->session_.tool_state->tools;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    // See docs/adr/007-thinking-disabled-by-default.md.
    inputs.enable_thinking = false;

    common_chat_params params;
    try {
        params = common_chat_templates_apply(impl_->loaded_.chat_templates.get(), inputs);
    } catch (const std::exception&) {
        return;
    }

    const int new_prev_len = static_cast<int>(params.prompt.size());
    commit_rendered_prompt(impl_->session_.prompt_state.committed_prompt_len, new_prev_len);
}

void clear_kv_cache(Model::Impl& impl) {
    if (impl.session_.ctx) {
        llama_memory_clear(llama_get_memory(impl.session_.ctx.get()), false);
    }
    impl.session_.prompt_state.committed_prompt_len = 0;
}

void note_history_append(Model::Impl& impl) noexcept {
    note_history_mutation(PromptHistoryMutation::Append, impl.session_.prompt_state.dirty,
                          impl.session_.prompt_state.committed_prompt_len);
}

void note_history_rewrite(Model::Impl& impl) noexcept {
    note_history_mutation(PromptHistoryMutation::Rewrite, impl.session_.prompt_state.dirty,
                          impl.session_.prompt_state.committed_prompt_len);
    clear_kv_cache(impl);
}

void note_history_reset(Model::Impl& impl) noexcept {
    note_history_mutation(PromptHistoryMutation::Reset, impl.session_.prompt_state.dirty,
                          impl.session_.prompt_state.committed_prompt_len);
    clear_kv_cache(impl);
}

} // namespace zoo::core
