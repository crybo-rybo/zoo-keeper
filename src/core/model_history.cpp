/**
 * @file model_history.cpp
 * @brief History and context bookkeeping for `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <llama.h>

namespace zoo::core {

void Model::set_system_prompt(std::string_view prompt) {
    Message sys_msg = Message::system(std::string(prompt));

    if (!messages_.empty() && messages_[0].role == Role::System) {
        estimated_tokens_ -= estimate_message_tokens(messages_[0]);
        messages_[0] = std::move(sys_msg);
    } else {
        messages_.insert(messages_.begin(), std::move(sys_msg));
    }

    estimated_tokens_ += estimate_message_tokens(messages_[0]);
    note_history_rewrite();
}

Expected<void> Model::add_message(MessageView message) {
    auto err = validate_role_sequence(messages_, message.role());
    if (!err) {
        return std::unexpected(err.error());
    }

    messages_.push_back(Message::from_view(message));
    estimated_tokens_ += estimate_message_tokens(messages_.back());
    note_history_append();
    trim_history_to_fit();
    return {};
}

HistorySnapshot Model::get_history() const {
    return HistorySnapshot{messages_};
}

void Model::clear_history() {
    messages_.clear();
    estimated_tokens_ = 0;
    note_history_reset();
}

void Model::replace_history(HistorySnapshot snapshot) {
    messages_ = std::move(snapshot.messages);
    estimated_tokens_ = 0;
    for (const auto& m : messages_) {
        estimated_tokens_ += estimate_message_tokens(m);
    }
    // Invalidate the rendered-prompt cache and reset the committed position so
    // the next generation re-renders from scratch, but intentionally skip
    // clear_kv_cache(): the caller is restoring a previously valid history, and
    // any stale KV entries will be overwritten when the next full prompt is
    // decoded starting at position 0.
    prompt_state_.dirty = true;
    prompt_state_.committed_prompt_len = 0;
}

HistorySnapshot Model::swap_history(HistorySnapshot snapshot) {
    HistorySnapshot previous{std::move(messages_)};
    replace_history(std::move(snapshot));
    return previous;
}

int Model::context_size() const noexcept {
    return model_config_.context_size;
}

int Model::estimated_tokens() const noexcept {
    return estimated_tokens_;
}

bool Model::is_context_exceeded() const noexcept {
    return estimated_tokens_ > model_config_.context_size;
}

int Model::estimate_tokens(std::string_view text) const {
    if (vocab_) {
        static_assert(sizeof(int) == sizeof(llama_token));
        const int32_t raw =
            llama_tokenize(vocab_, text.data(), text.length(), nullptr, 0, false, true);
        const int n = (raw < 0) ? -raw : raw;
        if (n > 0) {
            return n;
        }
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

int Model::estimate_message_tokens(const Message& message) const {
    int total = estimate_tokens(message.content) + kTemplateOverheadPerMessage;
    if (!message.tool_call_id.empty()) {
        total += estimate_tokens(message.tool_call_id);
    }
    for (const auto& tc : message.tool_calls) {
        total += estimate_tokens(tc.name);
        total += estimate_tokens(tc.id);
        total += estimate_tokens(tc.arguments_json);
    }
    return total;
}

void Model::trim_history_to_fit() {
    // Direct `Model` use no longer applies an implicit history budget. The
    // agent runtime owns retention policy and can scope/swap histories around
    // requests without forcing an extra trim policy here.
}

void Model::rollback_last_message() noexcept {
    if (messages_.empty()) {
        return;
    }

    estimated_tokens_ -= estimate_message_tokens(messages_.back());
    if (estimated_tokens_ < 0) {
        estimated_tokens_ = 0;
    }

    messages_.pop_back();
    note_history_rewrite();
}

} // namespace zoo::core
