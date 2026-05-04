/**
 * @file runtime_helpers.hpp
 * @brief Shared utilities for AgentRuntime implementation files.
 */

#pragma once

#include "backend.hpp"
#include "callback_dispatcher.hpp"
#include "request.hpp"
#include "zoo/core/types.hpp"
#include <chrono>
#include <functional>
#include <optional>
#include <string>
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

/// Scope-owned schema grammar activation for extraction requests.
class ScopedGrammarOverride {
  public:
    static Expected<ScopedGrammarOverride> activate(AgentBackend& backend,
                                                    const std::string& grammar,
                                                    std::function<void()> restore_callback) {
        if (!backend.set_schema_grammar(grammar)) {
            return std::unexpected(
                Error{ErrorCode::ExtractionFailed, "Failed to initialize schema grammar"});
        }
        return ScopedGrammarOverride(ScopeExit(std::move(restore_callback)));
    }

    ScopedGrammarOverride(const ScopedGrammarOverride&) = delete;
    ScopedGrammarOverride& operator=(const ScopedGrammarOverride&) = delete;
    ScopedGrammarOverride(ScopedGrammarOverride&&) noexcept = default;
    ScopedGrammarOverride& operator=(ScopedGrammarOverride&&) noexcept = default;

  private:
    explicit ScopedGrammarOverride(ScopeExit guard) : guard_(std::move(guard)) {}

    ScopeExit guard_;
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

/// Owns temporary or retained request history changes for one request.
class RequestHistoryScope {
  public:
    static Expected<RequestHistoryScope> enter(AgentBackend& backend, HistoryMode mode,
                                               const std::vector<Message>& messages,
                                               size_t max_retained_messages,
                                               std::string_view stateful_request_name) {
        RequestHistoryScope scope(backend, mode, max_retained_messages);
        if (mode == HistoryMode::Replace) {
            scope.original_history_ = swap_history(backend, messages);
            scope.active_ = true;
            return Expected<RequestHistoryScope>(std::move(scope));
        }

        if (messages.size() != 1u) {
            return std::unexpected(Error{ErrorCode::InvalidMessageSequence,
                                         "Stateful " + std::string(stateful_request_name) +
                                             " requests must include exactly one message"});
        }

        auto add_result = backend.add_message(messages.front().view());
        if (!add_result) {
            return std::unexpected(add_result.error());
        }
        scope.active_ = true;
        return Expected<RequestHistoryScope>(std::move(scope));
    }

    RequestHistoryScope(const RequestHistoryScope&) = delete;
    RequestHistoryScope& operator=(const RequestHistoryScope&) = delete;

    RequestHistoryScope(RequestHistoryScope&& other) noexcept
        : backend_(std::exchange(other.backend_, nullptr)), mode_(other.mode_),
          original_history_(std::move(other.original_history_)),
          max_retained_messages_(other.max_retained_messages_),
          active_(std::exchange(other.active_, false)) {}

    RequestHistoryScope& operator=(RequestHistoryScope&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        close();
        backend_ = std::exchange(other.backend_, nullptr);
        mode_ = other.mode_;
        original_history_ = std::move(other.original_history_);
        max_retained_messages_ = other.max_retained_messages_;
        active_ = std::exchange(other.active_, false);
        return *this;
    }

    ~RequestHistoryScope() {
        close();
    }

  private:
    RequestHistoryScope(AgentBackend& backend, HistoryMode mode, size_t max_retained_messages)
        : backend_(&backend), mode_(mode), max_retained_messages_(max_retained_messages) {}

    void close() {
        if (!active_ || backend_ == nullptr) {
            return;
        }
        active_ = false;
        if (mode_ == HistoryMode::Replace) {
            backend_->swap_history(std::move(*original_history_));
        } else {
            backend_->trim_history(max_retained_messages_);
        }
    }

    AgentBackend* backend_;
    HistoryMode mode_;
    std::optional<HistorySnapshot> original_history_;
    size_t max_retained_messages_;
    bool active_ = false;
};

/// Aggregates token usage and latency across one or more generation passes.
class GenerationStats {
  public:
    explicit GenerationStats(std::chrono::steady_clock::time_point start_time)
        : start_time_(start_time) {}

    void record_pass(std::chrono::steady_clock::time_point pass_start,
                     std::chrono::steady_clock::time_point pass_end,
                     bool first_token_received_this_pass,
                     std::chrono::steady_clock::time_point first_token_time_this_pass,
                     int prompt_tokens, int completion_tokens) {
        const bool had_first_token_before = first_token_received_;
        if (first_token_received_this_pass && !first_token_received_) {
            first_token_time_ = first_token_time_this_pass;
            first_token_received_ = true;
        }
        if (first_token_received_) {
            const auto interval_start =
                had_first_token_before ? pass_start : first_token_time_this_pass;
            generation_time_after_first_token_ += (pass_end - interval_start);
        }

        prompt_tokens_ += prompt_tokens;
        completion_tokens_ += completion_tokens;
    }

    [[nodiscard]] TokenUsage usage() const {
        return TokenUsage{
            prompt_tokens_,
            completion_tokens_,
            prompt_tokens_ + completion_tokens_,
        };
    }

    [[nodiscard]] Metrics metrics(std::chrono::steady_clock::time_point end_time) const {
        Metrics result;
        result.latency_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        if (first_token_received_) {
            result.time_to_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                first_token_time_ - start_time_);
            const auto generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                generation_time_after_first_token_);
            if (generation_ms.count() > 0) {
                result.tokens_per_second = (completion_tokens_ * 1000.0) / generation_ms.count();
            }
        }
        return result;
    }

  private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point first_token_time_;
    bool first_token_received_ = false;
    std::chrono::steady_clock::duration generation_time_after_first_token_{};
    int prompt_tokens_ = 0;
    int completion_tokens_ = 0;
};

struct GenerationPassResult {
    GenerationResult generation;
    int completion_tokens = 0;
};

/// Runs one model generation pass and records callback/token metrics.
class GenerationRunner {
  public:
    GenerationRunner(AgentBackend& backend, CallbackDispatcher& callback_dispatcher)
        : backend_(backend), callback_dispatcher_(callback_dispatcher) {}

    Expected<GenerationPassResult> run(const GenerationOptions& options,
                                       AsyncTokenCallback* streaming_callback,
                                       CancellationCallback should_cancel, GenerationStats& stats) {
        int completion_tokens = 0;
        const auto generation_start_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point first_token_time_this_pass;
        bool first_token_received_this_pass = false;

        auto callback = [&](std::string_view token) -> TokenAction {
            TokenAction action = TokenAction::Continue;
            if (streaming_callback != nullptr && *streaming_callback) {
                action = callback_dispatcher_.dispatch(*streaming_callback, token);
            }
            if (!first_token_received_this_pass) {
                first_token_time_this_pass = std::chrono::steady_clock::now();
                first_token_received_this_pass = true;
            }
            ++completion_tokens;
            return action;
        };

        auto generated =
            backend_.generate_from_history(options, TokenCallback(callback), should_cancel);
        callback_dispatcher_.drain();
        if (!generated) {
            return std::unexpected(generated.error());
        }

        stats.record_pass(generation_start_time, std::chrono::steady_clock::now(),
                          first_token_received_this_pass, first_token_time_this_pass,
                          generated->prompt_tokens, completion_tokens);
        return GenerationPassResult{std::move(*generated), completion_tokens};
    }

  private:
    AgentBackend& backend_;
    CallbackDispatcher& callback_dispatcher_;
};

} // namespace zoo::internal::agent
