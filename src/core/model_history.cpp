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

void Model::set_system_prompt(const std::string& prompt) {
    Message sys_msg = Message::system(prompt);

    if (!messages_.empty() && messages_[0].role == Role::System) {
        estimated_tokens_ -= estimate_tokens(messages_[0].content) + kTemplateOverheadPerMessage;
        messages_[0] = std::move(sys_msg);
    } else {
        messages_.insert(messages_.begin(), std::move(sys_msg));
    }

    estimated_tokens_ += estimate_tokens(prompt) + kTemplateOverheadPerMessage;
    note_history_rewrite();
}

Expected<void> Model::add_message(const Message& message) {
    auto err = validate_role_sequence(messages_, message.role);
    if (!err) {
        return std::unexpected(err.error());
    }

    messages_.push_back(message);
    estimated_tokens_ += estimate_tokens(message.content) + kTemplateOverheadPerMessage;
    note_history_append();
    trim_history_to_fit();
    return {};
}

std::vector<Message> Model::get_history() const {
    return messages_;
}

void Model::clear_history() {
    messages_.clear();
    estimated_tokens_ = 0;
    note_history_reset();
}

void Model::replace_messages(std::vector<Message> messages) {
    messages_ = std::move(messages);
    estimated_tokens_ = 0;
    for (const auto& m : messages_) {
        estimated_tokens_ += estimate_tokens(m.content) + kTemplateOverheadPerMessage;
    }
    // Invalidate the rendered-prompt cache and reset the committed position so
    // the next generation re-renders from scratch, but intentionally skip
    // clear_kv_cache(): the caller is restoring a previously valid history, and
    // any stale KV entries will be overwritten when the next full prompt is
    // decoded starting at position 0.
    prompt_state_.cached_messages_dirty = true;
    prompt_state_.committed_prompt_len = 0;
}

int Model::context_size() const noexcept {
    return config_.context_size;
}

int Model::estimated_tokens() const noexcept {
    return estimated_tokens_;
}

bool Model::is_context_exceeded() const noexcept {
    return estimated_tokens_ > config_.context_size;
}

int Model::estimate_tokens(const std::string& text) const {
    if (vocab_) {
        static_assert(sizeof(int) == sizeof(llama_token));
        const int32_t raw =
            llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, false, true);
        const int n = (raw < 0) ? -raw : raw;
        if (n > 0) {
            return n;
        }
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

void Model::trim_history_to_fit() {
    const size_t system_offset =
        (!messages_.empty() && messages_.front().role == Role::System) ? 1u : 0u;
    const size_t max_messages = config_.max_history_messages;

    if (messages_.size() <= system_offset + max_messages) {
        return;
    }

    size_t erase_end = messages_.size() - max_messages;
    if (erase_end < system_offset) {
        erase_end = system_offset;
    }

    while (erase_end < messages_.size() && messages_[erase_end].role != Role::User) {
        ++erase_end;
    }

    if (erase_end <= system_offset) {
        return;
    }

    for (size_t index = system_offset; index < erase_end; ++index) {
        estimated_tokens_ -=
            estimate_tokens(messages_[index].content) + kTemplateOverheadPerMessage;
    }
    if (estimated_tokens_ < 0) {
        estimated_tokens_ = 0;
    }

    messages_.erase(messages_.begin() + static_cast<std::ptrdiff_t>(system_offset),
                    messages_.begin() + static_cast<std::ptrdiff_t>(erase_end));
    note_history_rewrite();
}

void Model::rollback_last_message() noexcept {
    if (messages_.empty()) {
        return;
    }

    estimated_tokens_ -= estimate_tokens(messages_.back().content) + kTemplateOverheadPerMessage;
    if (estimated_tokens_ < 0) {
        estimated_tokens_ = 0;
    }

    messages_.pop_back();
    note_history_rewrite();
}

} // namespace zoo::core
