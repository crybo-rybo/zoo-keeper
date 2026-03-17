/**
 * @file runtime_helpers.hpp
 * @brief Shared utilities for AgentRuntime implementation files.
 */

#pragma once

#include "backend.hpp"
#include "zoo/core/types.hpp"
#include <functional>
#include <vector>

namespace zoo::internal::agent {

/// RAII guard that invokes a callback on scope exit.
class ScopeExit {
  public:
    explicit ScopeExit(std::function<void()> callback) : callback_(std::move(callback)) {}

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&&) noexcept = default;
    ScopeExit& operator=(ScopeExit&&) noexcept = default;

    ~ScopeExit() {
        if (callback_) {
            callback_();
        }
    }

  private:
    std::function<void()> callback_;
};

/// Replace backend history with the given messages. Returns error on failure.
inline Expected<void> load_history(AgentBackend& backend, const std::vector<Message>& messages) {
    backend.clear_history();
    for (const auto& message : messages) {
        if (auto add_result = backend.add_message(message); !add_result) {
            return std::unexpected(add_result.error());
        }
    }
    return {};
}

/// Restore backend history without a redundant KV-cache flush.
inline void restore_history(AgentBackend& backend, std::vector<Message> messages) {
    // Use replace_messages rather than clear_history() + add_message() to avoid
    // a redundant KV-cache flush.  The next generation will re-render from
    // position zero and naturally overwrite any stale entries left by the
    // scoped inference; entries beyond the restored prompt length are harmless
    // because causal attention never reaches them.
    backend.replace_messages(std::move(messages));
}

} // namespace zoo::internal::agent
