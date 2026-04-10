/**
 * @file runtime_helpers.hpp
 * @brief Shared utilities for AgentRuntime implementation files.
 */

#pragma once

#include "backend.hpp"
#include "zoo/core/types.hpp"
#include <functional>
#include <utility>
#include <vector>

namespace zoo::internal::agent {

/// RAII guard that invokes a callback on scope exit.
class ScopeExit {
  public:
    explicit ScopeExit(std::function<void()> callback) : callback_(std::move(callback)) {}

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&& other) noexcept : callback_(std::exchange(other.callback_, {})) {}
    ScopeExit& operator=(ScopeExit&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        if (callback_) {
            callback_();
        }
        callback_ = std::exchange(other.callback_, {});
        return *this;
    }

    ~ScopeExit() {
        if (callback_) {
            callback_();
        }
    }

  private:
    std::function<void()> callback_;
};

/// Replace backend history with the given messages. Returns error on failure.
inline HistorySnapshot snapshot_from_messages(const std::vector<Message>& messages) {
    HistorySnapshot snapshot;
    snapshot.messages = messages;
    return snapshot;
}

/// Replace backend history while returning the previous snapshot.
inline HistorySnapshot swap_history(AgentBackend& backend, const std::vector<Message>& messages) {
    return backend.swap_history(snapshot_from_messages(messages));
}

} // namespace zoo::internal::agent
