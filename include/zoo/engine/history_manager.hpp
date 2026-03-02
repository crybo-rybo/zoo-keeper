#pragma once

#include "../types.hpp"
#include <vector>
#include <algorithm>
#include <string_view>
#include <functional>
#include <mutex>

namespace zoo {
namespace engine {

/**
 * @brief Manages conversation history with validation and token estimation
 *
 * MVP Responsibilities:
 * - Store conversation messages in order
 * - Basic role sequence validation
 * - Token count estimation (4 chars ≈ 1 token, or pluggable tokenizer)
 * - Provide full history for rendering
 *
 * MVP Shortcuts (Production in Phase 2-3):
 * - No KV cache reuse optimization
 * - Basic role validation instead of full state machine
 *
 * Thread Safety: Internally synchronized via mutex
 */
class HistoryManager {
public:
    /**
     * @brief Construct with context size limit and optional tokenizer
     *
     * @param context_size Maximum context window size in tokens
     * @param tokenizer Optional callable for accurate token counting.
     *        Receives text and returns token count (> 0). When null or
     *        when the callable returns <= 0, falls back to the 4-chars-per-token
     *        heuristic.
     * @param template_overhead_per_message Additional tokens added per message
     *        by the chat template (role markers, BOS/EOS, turn separators).
     *        Default of 8 covers most models (Gemma, Llama, Phi add ~6-10
     *        special tokens per turn).
     */
    explicit HistoryManager(
        int context_size,
        std::function<int(const std::string&)> tokenizer = nullptr,
        int template_overhead_per_message = 8
    )
        : context_size_(context_size)
        , estimated_tokens_(0)
        , tokenizer_(std::move(tokenizer))
        , template_overhead_per_message_(template_overhead_per_message)
    {}

    /**
     * @brief Add a message to history
     *
     * Validates role sequence and updates token estimate.
     *
     * @param message Message to add
     * @return Expected<void> Success or validation error
     */
    Expected<void> add_message(const Message& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Basic role validation
        if (auto err = validate_role_sequence(message.role); !err) {
            return tl::unexpected(err.error());
        }

        // Add to history
        messages_.push_back(message);

        // Update token estimate (content tokens + chat template overhead per message)
        estimated_tokens_ += estimate_tokens(message.content) + template_overhead_per_message_;

        return {};
    }

    /**
     * @brief Add a message to history (move overload)
     *
     * Validates role sequence and moves the message into history.
     *
     * @param message Message to move into history
     * @return Expected<void> Success or validation error
     */
    Expected<void> add_message(Message&& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Basic role validation
        if (auto err = validate_role_sequence(message.role); !err) {
            return tl::unexpected(err.error());
        }

        // Update token estimate before moving (content tokens + chat template overhead per message)
        estimated_tokens_ += estimate_tokens(message.content) + template_overhead_per_message_;

        // Move into history
        messages_.push_back(std::move(message));

        return {};
    }

    /**
     * @brief Set system prompt
     *
     * Replaces or adds system message at the beginning of history.
     *
     * @param prompt System prompt content
     */
    void set_system_prompt(const std::string& prompt) {
        std::lock_guard<std::mutex> lock(mutex_);

        Message sys_msg = Message::system(prompt);

        // Replace existing system prompt or insert at beginning
        if (!messages_.empty() && messages_[0].role == Role::System) {
            // Subtract old content tokens + overhead
            estimated_tokens_ -= estimate_tokens(messages_[0].content) + template_overhead_per_message_;
            messages_[0] = sys_msg;
        } else {
            messages_.insert(messages_.begin(), sys_msg);
        }

        // Add new content tokens + overhead
        estimated_tokens_ += estimate_tokens(prompt) + template_overhead_per_message_;
    }

    /**
     * @brief Get a copy of all messages
     *
     * Returns by value so the caller holds an independent snapshot.
     * The mutex is released before the copy is returned, so callers
     * do not hold the lock while working with the messages.
     *
     * @return std::vector<Message> Copy of message history
     */
    std::vector<Message> get_messages() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return messages_;
    }

    /**
     * @brief Get estimated token count
     *
     * Uses pluggable tokenizer when available, otherwise 4 chars per token.
     *
     * @return int Estimated token count
     */
    int get_estimated_tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return estimated_tokens_;
    }

    /**
     * @brief Check if context window is exceeded
     *
     * @return bool True if estimated tokens exceed context size
     */
    bool is_context_exceeded() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return estimated_tokens_ > context_size_;
    }

    /**
     * @brief Remove the last message from history
     *
     * Used for error recovery when generation fails after adding
     * a user message. Updates token estimate accordingly.
     *
     * @return bool True if a message was removed, false if history was empty
     */
    bool remove_last_message() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (messages_.empty()) {
            return false;
        }
        estimated_tokens_ -= estimate_tokens(messages_.back().content) + template_overhead_per_message_;
        messages_.pop_back();
        return true;
    }

    /**
     * @brief Remove oldest non-system messages until token budget is met.
     *
     * This is used by long-context memory integration to keep recent turns in
     * active history while archiving older turns externally.
     *
     * @param target_tokens Desired post-prune token budget
     * @param min_messages_to_keep Minimum number of newest messages to keep
     * @return std::vector<Message> Removed messages in original order
     */
    std::vector<Message> prune_oldest_messages_until(
        int target_tokens,
        size_t min_messages_to_keep = 4
    ) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<Message> removed;
        if (messages_.empty()) {
            return removed;
        }

        const bool has_system = messages_[0].role == Role::System;
        const size_t first_removable_index = has_system ? 1U : 0U;

        // Count how many messages to remove without modifying the vector,
        // then erase them in a single batch to avoid O(n^2) shifts.
        size_t remove_count = 0;
        int tokens_to_subtract = 0;
        while (estimated_tokens_ - tokens_to_subtract > target_tokens) {
            if (messages_.size() - remove_count <= first_removable_index + min_messages_to_keep) {
                break;
            }
            size_t idx = first_removable_index + remove_count;
            tokens_to_subtract += estimate_tokens(messages_[idx].content) + template_overhead_per_message_;
            ++remove_count;
        }

        if (remove_count > 0) {
            auto begin = messages_.begin() + static_cast<std::ptrdiff_t>(first_removable_index);
            auto end = begin + static_cast<std::ptrdiff_t>(remove_count);
            removed.reserve(remove_count);
            for (auto it = begin; it != end; ++it) {
                removed.push_back(std::move(*it));
            }
            messages_.erase(begin, end);
            estimated_tokens_ -= tokens_to_subtract;
        }

        return removed;
    }

    /**
     * @brief Restore previously pruned messages near the front of history.
     *
     * Messages are inserted after the system prompt when present.
     */
    void prepend_messages(const std::vector<Message>& messages) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (messages.empty()) {
            return;
        }

        const bool has_system = !messages_.empty() && messages_[0].role == Role::System;
        const auto insert_pos = messages_.begin() + static_cast<std::ptrdiff_t>(has_system ? 1U : 0U);
        messages_.insert(insert_pos, messages.begin(), messages.end());

        for (const auto& msg : messages) {
            estimated_tokens_ += estimate_tokens(msg.content) + template_overhead_per_message_;
        }
    }

    /**
     * @brief Clear all messages
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        messages_.clear();
        estimated_tokens_ = 0;
    }

    /**
     * @brief Get context size
     */
    int get_context_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return context_size_;
    }

    /**
     * @brief Synchronize the internal token estimate with actual KV cache usage.
     *
     * Call this after tokenization to keep estimates calibrated against real
     * token counts rather than relying solely on the heuristic estimate.
     * Only updates if actual_total is positive (ignores zero/negative values).
     *
     * @param actual_total Actual total tokens used (e.g., kv_cache_used + prompt_delta_tokens)
     */
    void sync_token_estimate(int actual_total) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (actual_total > 0) {
            estimated_tokens_ = actual_total;
        }
    }

private:
    mutable std::mutex mutex_;
    std::vector<Message> messages_;
    int context_size_;
    int estimated_tokens_;
    std::function<int(const std::string&)> tokenizer_;
    int template_overhead_per_message_;

    /**
     * @brief Estimate token count from text
     *
     * Uses the pluggable tokenizer when available and returns a valid count.
     * Falls back to the 4-chars-per-token heuristic.
     *
     * @param text Input text
     * @return int Estimated token count (>= 1)
     */
    int estimate_tokens(const std::string& text) const {
        if (tokenizer_) {
            int count = tokenizer_(text);
            if (count > 0) return count;
        }
        return std::max(1, static_cast<int>(text.length() / 4));
    }

    /**
     * @brief Validate role sequence
     *
     * MVP: Basic checks
     * - User/Assistant alternation (after first message)
     * - System only at start
     * - No consecutive roles (except tool calls)
     *
     * Production: Full state machine per TRD spec
     *
     * @param role Role to add
     * @return Expected<void> Success or error
     *
     * NOTE: Must be called with mutex_ already held.
     */
    Expected<void> validate_role_sequence(Role role) const {
        // Empty history: allow any role except Tool
        if (messages_.empty()) {
            if (role == Role::Tool) {
                return tl::unexpected(Error{
                    ErrorCode::InvalidMessageSequence,
                    "First message cannot be a tool response"
                });
            }
            return {};
        }

        // System messages only allowed at start
        if (role == Role::System) {
            // Allow only if replacing first system message (handled by set_system_prompt)
            return tl::unexpected(Error{
                ErrorCode::InvalidMessageSequence,
                "System message only allowed at the beginning"
            });
        }

        // Check for consecutive same roles (simplified validation)
        const Role last_role = messages_.back().role;
        if (role == last_role && role != Role::Tool) {
            return tl::unexpected(Error{
                ErrorCode::InvalidMessageSequence,
                "Cannot have consecutive messages with the same role (except Tool)"
            });
        }

        return {};
    }
};

} // namespace engine
} // namespace zoo
