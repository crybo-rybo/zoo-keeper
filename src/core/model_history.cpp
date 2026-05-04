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

    if (!impl_->session_.messages.empty() && impl_->session_.messages[0].role == Role::System) {
        impl_->session_.estimated_tokens -=
            estimate_message_tokens(*impl_, impl_->session_.messages[0]);
        impl_->session_.messages[0] = std::move(sys_msg);
    } else {
        impl_->session_.messages.insert(impl_->session_.messages.begin(), std::move(sys_msg));
    }

    impl_->session_.estimated_tokens +=
        estimate_message_tokens(*impl_, impl_->session_.messages[0]);
    note_history_rewrite(*impl_);
}

Expected<void> Model::add_message(MessageView message) {
    auto err = validate_role_sequence(impl_->session_.messages, message.role());
    if (!err) {
        return std::unexpected(err.error());
    }

    impl_->session_.messages.push_back(Message::from_view(message));
    impl_->session_.estimated_tokens +=
        estimate_message_tokens(*impl_, impl_->session_.messages.back());
    note_history_append(*impl_);
    trim_history_to_fit(*impl_);
    return {};
}

HistorySnapshot Model::get_history() const {
    return HistorySnapshot{impl_->session_.messages};
}

void Model::clear_history() {
    impl_->session_.messages.clear();
    impl_->session_.estimated_tokens = 0;
    note_history_reset(*impl_);
}

void Model::replace_history(HistorySnapshot snapshot) {
    impl_->session_.messages = std::move(snapshot.messages);
    impl_->session_.estimated_tokens = 0;
    for (const auto& m : impl_->session_.messages) {
        impl_->session_.estimated_tokens += estimate_message_tokens(*impl_, m);
    }
    note_history_rewrite(*impl_);
}

HistorySnapshot Model::swap_history(HistorySnapshot snapshot) {
    HistorySnapshot previous{std::move(impl_->session_.messages)};
    replace_history(std::move(snapshot));
    return previous;
}

int Model::context_size() const noexcept {
    return impl_->loaded_.model_config.context_size;
}

int Model::estimated_tokens() const noexcept {
    return impl_->session_.estimated_tokens;
}

bool Model::is_context_exceeded() const noexcept {
    return impl_->session_.estimated_tokens > impl_->loaded_.model_config.context_size;
}

int estimate_tokens(const Model::Impl& impl, std::string_view text) {
    if (impl.loaded_.vocab) {
        static_assert(sizeof(int) == sizeof(llama_token));
        const int32_t raw =
            llama_tokenize(impl.loaded_.vocab, text.data(), text.length(), nullptr, 0, false, true);
        const int n = (raw < 0) ? -raw : raw;
        if (n > 0) {
            return n;
        }
    }
    return std::max(1, static_cast<int>(text.length() / 4));
}

int estimate_message_tokens(const Model::Impl& impl, const Message& message) {
    int total = estimate_tokens(impl, message.content) + Model::Impl::kTemplateOverheadPerMessage;
    if (!message.tool_call_id.empty()) {
        total += estimate_tokens(impl, message.tool_call_id);
    }
    for (const auto& tc : message.tool_calls) {
        total += estimate_tokens(impl, tc.name);
        total += estimate_tokens(impl, tc.id);
        total += estimate_tokens(impl, tc.arguments_json);
    }
    return total;
}

void Model::trim_history(size_t max_non_system_messages) {
    const size_t system_offset =
        (!impl_->session_.messages.empty() && impl_->session_.messages.front().role == Role::System)
            ? 1u
            : 0u;

    if (impl_->session_.messages.size() <= system_offset + max_non_system_messages) {
        return;
    }

    size_t erase_end = impl_->session_.messages.size() - max_non_system_messages;

    // Align to a user-message boundary so we don't start mid-exchange.
    while (erase_end < impl_->session_.messages.size() &&
           impl_->session_.messages[erase_end].role != Role::User) {
        ++erase_end;
    }

    for (size_t index = system_offset; index < erase_end; ++index) {
        impl_->session_.estimated_tokens -=
            estimate_message_tokens(*impl_, impl_->session_.messages[index]);
    }
    if (impl_->session_.estimated_tokens < 0) {
        impl_->session_.estimated_tokens = 0;
    }

    impl_->session_.messages.erase(
        impl_->session_.messages.begin() + static_cast<std::ptrdiff_t>(system_offset),
        impl_->session_.messages.begin() + static_cast<std::ptrdiff_t>(erase_end));
    note_history_rewrite(*impl_);
}

void trim_history_to_fit(Model::Impl&) {
    // Called from add_message() — no longer applies an implicit budget.
    // The agent runtime owns retention policy via trim_history().
}

void rollback_last_message(Model::Impl& impl) noexcept {
    if (impl.session_.messages.empty()) {
        return;
    }

    impl.session_.estimated_tokens -= estimate_message_tokens(impl, impl.session_.messages.back());
    if (impl.session_.estimated_tokens < 0) {
        impl.session_.estimated_tokens = 0;
    }

    impl.session_.messages.pop_back();
    note_history_rewrite(impl);
}

} // namespace zoo::core
