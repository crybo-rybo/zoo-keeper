#pragma once

#include "../types.hpp"
#include <vector>
#include <algorithm>

namespace zoo {
namespace engine {

/**
 * @brief Manages conversation history with validation and token estimation
 *
 * MVP Responsibilities:
 * - Store conversation messages in order
 * - Basic role sequence validation
 * - Token count estimation (4 chars ≈ 1 token)
 * - Provide full history for rendering
 *
 * MVP Shortcuts (Production in Phase 2-3):
 * - No context pruning (will overflow eventually)
 * - No KV cache reuse optimization
 * - Simple token estimation instead of real tokenization
 * - Basic role validation instead of full state machine
 *
 * Thread Safety: Single-threaded (inference thread only)
 */
class HistoryManager {
public:
    /**
     * @brief Construct with context size limit
     *
     * @param context_size Maximum context window size in tokens
     */
    explicit HistoryManager(int context_size)
        : context_size_(context_size)
        , estimated_tokens_(0)
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
     * @brief Set system prompt
     *
     * Replaces or adds system message at the beginning of history.
     *
     * @param prompt System prompt content
     */
    void set_system_prompt(const std::string& prompt) {
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
     * @return const std::vector<Message>& Message history
     */
    const std::vector<Message>& get_messages() const {
        return messages_;
    }

    /**
     * @brief Get estimated token count
     *
     * Uses 4 chars ≈ 1 token heuristic.
     *
     * @return int Estimated token count
     */
    int get_estimated_tokens() const {
        return estimated_tokens_;
    }

    /**
     * @brief Check if context window is exceeded
     *
     * MVP: No pruning, just warning.
     *
     * @return bool True if estimated tokens exceed context size
     */
    bool is_context_exceeded() const {
        return estimated_tokens_ > context_size_;
    }

    /**
     * @brief Clear all messages
     */
    void clear() {
        messages_.clear();
        estimated_tokens_ = 0;
    }

    /**
     * @brief Get context size
     */
    int get_context_size() const {
        return context_size_;
    }

private:
    std::vector<Message> messages_;
    int context_size_;
    int estimated_tokens_;

    /**
     * @brief Estimate token count from text
     *
     * Simple heuristic: ~4 characters per token
     * Production: Use actual backend tokenization with caching
     *
     * @param text Input text
     * @return int Estimated token count
     */
    static int estimate_tokens(const std::string& text) {
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
        if (role == Role::System && !messages_.empty()) {
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
