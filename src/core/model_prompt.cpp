/**
 * @file model_prompt.cpp
 * @brief Incremental prompt rendering and KV-cache bookkeeping for `Model`.
 */

#include "zoo/core/model.hpp"

#include "zoo/internal/core/prompt_bookkeeping.hpp"

#include <llama.h>

namespace zoo::core {

const std::vector<llama_chat_message>& Model::llama_messages() {
    if (!prompt_state_.cached_messages_dirty) {
        return prompt_state_.cached_llama_messages;
    }

    prompt_state_.cached_llama_messages.clear();
    prompt_state_.cached_llama_messages.reserve(messages_.size());
    for (const auto& msg : messages_) {
        prompt_state_.cached_llama_messages.push_back({role_to_string(msg.role), msg.content.c_str()});
    }
    prompt_state_.cached_messages_dirty = false;
    return prompt_state_.cached_llama_messages;
}

Expected<std::string> Model::render_prompt_delta() {
    const auto& llama_msgs = llama_messages();
    int new_len =
        llama_chat_apply_template(tmpl_, llama_msgs.data(), llama_msgs.size(), true, nullptr, 0);

    if (new_len < 0) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "llama_chat_apply_template failed"});
    }

    if (new_len == 0) {
        return std::string{};
    }

    if (new_len > static_cast<int>(prompt_state_.formatted_prompt.size())) {
        prompt_state_.formatted_prompt.resize(static_cast<size_t>(new_len));
    }

    new_len = llama_chat_apply_template(tmpl_, llama_msgs.data(), llama_msgs.size(), true,
                                        prompt_state_.formatted_prompt.data(),
                                        prompt_state_.formatted_prompt.size());
    if (new_len < 0) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "llama_chat_apply_template failed"});
    }

    if (rendered_prompt_requires_kv_reset(prompt_state_.committed_prompt_len, new_len)) {
        clear_kv_cache();
    }

    return std::string(prompt_state_.formatted_prompt.begin() + prompt_state_.committed_prompt_len,
                       prompt_state_.formatted_prompt.begin() + new_len);
}

void Model::finalize_response() {
    const auto& llama_msgs = llama_messages();
    int new_prev_len =
        llama_chat_apply_template(tmpl_, llama_msgs.data(), llama_msgs.size(), false, nullptr, 0);
    commit_rendered_prompt(prompt_state_.committed_prompt_len, new_prev_len);
}

void Model::clear_kv_cache() {
    if (ctx_) {
        llama_memory_clear(llama_get_memory(ctx_.get()), false);
    }
    prompt_state_.committed_prompt_len = 0;
}

void Model::note_history_append() noexcept {
    note_history_mutation(PromptHistoryMutation::Append, prompt_state_.cached_messages_dirty,
                          prompt_state_.committed_prompt_len);
}

void Model::note_history_rewrite() noexcept {
    note_history_mutation(PromptHistoryMutation::Rewrite, prompt_state_.cached_messages_dirty,
                          prompt_state_.committed_prompt_len);
    clear_kv_cache();
}

void Model::note_history_reset() noexcept {
    note_history_mutation(PromptHistoryMutation::Reset, prompt_state_.cached_messages_dirty,
                          prompt_state_.committed_prompt_len);
    clear_kv_cache();
}

} // namespace zoo::core
