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
     */
    explicit HistoryManager(
        int context_size,
        std::function<int(const std::string&)> tokenizer = nullptr
    )
        : context_size_(context_size)
        , estimated_tokens_(0)
        , tokenizer_(std::move(tokenizer))
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

        // Update token estimate
        estimated_tokens_ += estimate_tokens(message.content);

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

        // Update token estimate before moving
        estimated_tokens_ += estimate_tokens(message.content);

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
            estimated_tokens_ -= estimate_tokens(messages_[0].content);
            messages_[0] = sys_msg;
        } else {
            messages_.insert(messages_.begin(), sys_msg);
        }

        estimated_tokens_ += estimate_tokens(prompt);
    }

    /**
     * @brief Get all messages
     *
     * @return std::vector<Message> Copy of message history
     */
    const std::vector<Message>& get_messages() const {
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
        estimated_tokens_ -= estimate_tokens(messages_.back().content);
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

        while (estimated_tokens_ > target_tokens) {
            if (messages_.size() <= first_removable_index + min_messages_to_keep) {
                break;
            }

            Message removed_msg = std::move(messages_[first_removable_index]);
            estimated_tokens_ -= estimate_tokens(removed_msg.content);
            messages_.erase(messages_.begin() + static_cast<std::ptrdiff_t>(first_removable_index));
            removed.push_back(std::move(removed_msg));
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
            estimated_tokens_ += estimate_tokens(msg.content);
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

private:
    mutable std::mutex mutex_;
    std::vector<Message> messages_;
    int context_size_;
    int estimated_tokens_;
    std::function<int(const std::string&)> tokenizer_;

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
