/**
 * @file model_history.cpp
 * @brief History and context bookkeeping for `zoo::core::Model`.
 */

#include "core/model_impl.hpp"
#include "zoo/core/model.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <llama.h>

namespace zoo::core {

void Model::set_system_prompt(std::string_view prompt) {
    Message sys_msg = Message::system(std::string(prompt));

    if (!impl_->messages_.empty() && impl_->messages_[0].role == Role::System) {
        impl_->estimated_tokens_ -= estimate_message_tokens(impl_->messages_[0]);
        impl_->messages_[0] = std::move(sys_msg);
    } else {
        impl_->messages_.insert(impl_->messages_.begin(), std::move(sys_msg));
    }

    impl_->estimated_tokens_ += estimate_message_tokens(impl_->messages_[0]);
    note_history_rewrite();
}

Expected<void> Model::add_message(MessageView message) {
    auto err = validate_role_sequence(impl_->messages_, message.role());
    if (!err) {
        return std::unexpected(err.error());
    }

    impl_->messages_.push_back(Message::from_view(message));
    impl_->estimated_tokens_ += estimate_message_tokens(impl_->messages_.back());
    note_history_append();
    trim_history_to_fit();
    return {};
}

HistorySnapshot Model::get_history() const {
    return HistorySnapshot{impl_->messages_};
}

void Model::clear_history() {
    impl_->messages_.clear();
    impl_->estimated_tokens_ = 0;
    note_history_reset();
}

void Model::replace_history(HistorySnapshot snapshot) {
    impl_->messages_ = std::move(snapshot.messages);
    impl_->estimated_tokens_ = 0;
    for (const auto& m : impl_->messages_) {
        impl_->estimated_tokens_ += estimate_message_tokens(m);
    }
    // Invalidate the rendered-prompt cache and reset the committed position so
    // the next generation re-renders from scratch, but intentionally skip
    // clear_kv_cache(): the caller is restoring a previously valid history, and
    // any stale KV entries will be overwritten when the next full prompt is
    // decoded starting at position 0.
    impl_->prompt_state_.dirty = true;
    impl_->prompt_state_.committed_prompt_len = 0;
}

HistorySnapshot Model::swap_history(HistorySnapshot snapshot) {
    HistorySnapshot previous{std::move(impl_->messages_)};
    replace_history(std::move(snapshot));
    return previous;
}

int Model::context_size() const noexcept {
    return impl_->model_config_.context_size;
}

int Model::estimated_tokens() const noexcept {
    return impl_->estimated_tokens_;
}

bool Model::is_context_exceeded() const noexcept {
    return impl_->estimated_tokens_ > impl_->model_config_.context_size;
}

int Model::estimate_tokens(std::string_view text) const {
    if (impl_->vocab_) {
        static_assert(sizeof(int) == sizeof(llama_token));
        const int32_t raw =
            llama_tokenize(impl_->vocab_, text.data(), text.length(), nullptr, 0, false, true);
        const int n = (raw < 0) ? -raw : raw;
        if (n > 0) {
            return n;
        }
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

int Model::estimate_message_tokens(const Message& message) const {
    int total = estimate_tokens(message.content) + Impl::kTemplateOverheadPerMessage;
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

void Model::trim_history(size_t max_non_system_messages) {
    const size_t system_offset =
        (!impl_->messages_.empty() && impl_->messages_.front().role == Role::System) ? 1u : 0u;

    if (impl_->messages_.size() <= system_offset + max_non_system_messages) {
        return;
    }

    size_t erase_end = impl_->messages_.size() - max_non_system_messages;
    if (erase_end < system_offset) {
        erase_end = system_offset;
    }

    // Align to a user-message boundary so we don't start mid-exchange.
    while (erase_end < impl_->messages_.size() && impl_->messages_[erase_end].role != Role::User) {
        ++erase_end;
    }

    if (erase_end <= system_offset) {
        return;
    }

    for (size_t index = system_offset; index < erase_end; ++index) {
        impl_->estimated_tokens_ -= estimate_message_tokens(impl_->messages_[index]);
    }
    if (impl_->estimated_tokens_ < 0) {
        impl_->estimated_tokens_ = 0;
    }

    impl_->messages_.erase(impl_->messages_.begin() + static_cast<std::ptrdiff_t>(system_offset),
                           impl_->messages_.begin() + static_cast<std::ptrdiff_t>(erase_end));
    note_history_rewrite();
}

void Model::trim_history_to_fit() {
    // Called from add_message() — no longer applies an implicit budget.
    // The agent runtime owns retention policy via trim_history().
}

void Model::rollback_last_message() noexcept {
    if (impl_->messages_.empty()) {
        return;
    }

    impl_->estimated_tokens_ -= estimate_message_tokens(impl_->messages_.back());
    if (impl_->estimated_tokens_ < 0) {
        impl_->estimated_tokens_ = 0;
    }

    impl_->messages_.pop_back();
    note_history_rewrite();
}

} // namespace zoo::core
