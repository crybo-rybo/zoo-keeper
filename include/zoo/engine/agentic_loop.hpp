#pragma once

#include "../types.hpp"
#include "../backend/IBackend.hpp"
#include "history_manager.hpp"
#include "request_queue.hpp"
#include <atomic>
#include <memory>
#include <chrono>
#include <mutex>

namespace zoo {
namespace engine {

/**
 * @brief Core inference loop that processes requests from the queue
 *
 * The AgenticLoop runs on the inference thread and:
 * - Pops requests from the RequestQueue
 * - Formats prompts via backend->format_prompt() (incremental)
 * - Tokenizes prompts and calls backend->generate()
 * - Adds assistant responses to HistoryManager
 * - Calls backend->finalize_response() to update prompt cache
 * - Tracks metrics (latency, TTFT, tokens/sec)
 * - Supports cancellation via atomic flag
 *
 * MVP Scope:
 * - Single-turn text generation (no tool calling)
 * - Streaming token callbacks
 * - Basic error handling
 * - Performance metrics
 *
 * Thread Safety:
 * - Designed to run on a single inference thread
 * - Cancellation flag is atomic for cross-thread signaling
 * - RequestQueue handles thread-safe communication
 */
class AgenticLoop {
public:
    /**
     * @brief Construct agentic loop with dependencies
     *
     * @param backend Backend implementation (LlamaBackend or MockBackend)
     * @param history History manager instance
     * @param config Agent configuration
     * @param history_mutex Mutex for synchronizing history access (optional, for thread safety)
     */
    AgenticLoop(
        std::shared_ptr<backend::IBackend> backend,
        std::shared_ptr<HistoryManager> history,
        const Config& config,
        std::mutex* history_mutex = nullptr
    )
        : backend_(std::move(backend))
        , history_(std::move(history))
        , config_(config)
        , history_mutex_(history_mutex)
        , cancelled_(false)
    {}

    /**
     * @brief Process a single request
     *
     * This is the core inference method:
     * 1. Add user message to history
     * 2. Check context window
     * 3. Render conversation history
     * 4. Tokenize prompt
     * 5. Generate completion with streaming
     * 6. Add assistant response to history
     * 7. Track metrics
     *
     * @param request Request to process
     * @return Expected<Response> Generated response or error
     */
    Expected<Response> process_request(const Request& request) {
        // Check cancellation
        if (cancelled_.load(std::memory_order_acquire)) {
            return tl::unexpected(Error{
                ErrorCode::RequestCancelled,
                "Request cancelled"
            });
        }

        auto start_time = std::chrono::steady_clock::now();

        // Prepare prompt with history locked to prevent races with user thread
        std::string prompt;
        {
            auto lock = lock_history();

            // 1. Add user message to history
            if (auto result = history_->add_message(request.message); !result) {
                return tl::unexpected(result.error());
            }

            // 2. Check context window (MVP: warning only, no pruning)
            if (history_->is_context_exceeded()) {
                // Rollback user message before returning error
                history_->remove_last_message();
                return tl::unexpected(Error{
                    ErrorCode::ContextWindowExceeded,
                    "Context window exceeded",
                    "Estimated tokens: " + std::to_string(history_->get_estimated_tokens()) +
                    " / " + std::to_string(history_->get_context_size())
                });
            }

            // 3. Render conversation history
            auto prompt_result = backend_->format_prompt(history_->get_messages());
            if (!prompt_result) {
                // Rollback user message before returning error
                history_->remove_last_message();
                return tl::unexpected(prompt_result.error());
            }
            prompt = *prompt_result;
            // Lock released here - we no longer need to access history until after generation
        }

        // 4. Tokenize prompt
        auto tokens_result = backend_->tokenize(prompt);
        if (!tokens_result) {
            // Rollback user message from history on failure
            rollback_last_message();
            return tl::unexpected(tokens_result.error());
        }
        const auto& prompt_tokens = *tokens_result;

        // 5. Generate completion with streaming
        std::chrono::steady_clock::time_point first_token_time;
        bool first_token_received = false;
        int completion_tokens = 0;

        // Wrap streaming callback to track TTFT and token count
        std::optional<std::function<void(std::string_view)>> wrapped_callback;
        if (request.streaming_callback) {
            wrapped_callback = [&, user_callback = *request.streaming_callback](std::string_view token) {
                // Track first token time
                if (!first_token_received) {
                    first_token_time = std::chrono::steady_clock::now();
                    first_token_received = true;
                }

                // Increment token count
                ++completion_tokens;

                // Call user callback
                user_callback(token);
            };
        } else {
            // No user callback, but still track metrics
            wrapped_callback = [&](std::string_view) {
                if (!first_token_received) {
                    first_token_time = std::chrono::steady_clock::now();
                    first_token_received = true;
                }
                ++completion_tokens;
            };
        }

        // Generate
        auto generate_result = backend_->generate(
            prompt_tokens,
            config_.max_tokens,
            config_.stop_sequences,
            wrapped_callback
        );

        if (!generate_result) {
            // Rollback user message from history on failure
            rollback_last_message();
            return tl::unexpected(generate_result.error());
        }
        const std::string& generated_text = *generate_result;

        auto end_time = std::chrono::steady_clock::now();

        // 6. Add assistant response to history (with mutex protection)
        {
            auto lock = lock_history();

            Message assistant_msg = Message::assistant(generated_text);
            if (auto result = history_->add_message(std::move(assistant_msg)); !result) {
                return tl::unexpected(result.error());
            }

            // Update backend's prompt cache state (prev_len)
            backend_->finalize_response(history_->get_messages());
        }

        // 7. Build response with metrics
        Response response;
        response.text = generated_text;

        // Token usage
        response.usage.prompt_tokens = static_cast<int>(prompt_tokens.size());
        response.usage.completion_tokens = completion_tokens;
        response.usage.total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens;

        // Metrics
        auto total_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        );
        response.metrics.latency_ms = total_latency;

        if (first_token_received) {
            response.metrics.time_to_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                first_token_time - start_time
            );

            // Calculate tokens/sec for completion only
            auto generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - first_token_time
            );
            if (generation_time.count() > 0) {
                response.metrics.tokens_per_second =
                    (completion_tokens * 1000.0) / generation_time.count();
            }
        }

        return response;
    }

    /**
     * @brief Run the main inference loop
     *
     * Processes requests from the queue until shutdown.
     * This method is intended to run on the inference thread.
     *
     * @param queue Request queue to process
     */
    void run(RequestQueue& queue) {
        while (!cancelled_.load(std::memory_order_acquire)) {
            // Pop request (blocking)
            auto request_opt = queue.pop();

            // Check shutdown
            if (!request_opt) {
                break;  // Queue shutdown
            }

            // Process request
            auto result = process_request(*request_opt);

            // Note: In the Agent class, we'll use promises to return results
            // This method is a helper for the main loop structure
            (void)result;  // Result handled by caller (Agent class)
        }
    }

    /**
     * @brief Cancel ongoing operations
     *
     * Sets atomic flag to stop processing.
     * Thread-safe: Can be called from any thread.
     */
    void cancel() {
        cancelled_.store(true, std::memory_order_release);
    }

    /**
     * @brief Check if loop is cancelled
     *
     * @return bool True if cancelled
     */
    bool is_cancelled() const {
        return cancelled_.load(std::memory_order_acquire);
    }

    /**
     * @brief Reset cancellation flag
     */
    void reset() {
        cancelled_.store(false, std::memory_order_release);
    }

private:
    /**
     * @brief Rollback the last message from history
     *
     * Used for error recovery when tokenization or generation fails
     * after a user message has been added to history. Prevents the
     * history from being left in an inconsistent state (consecutive
     * same-role messages).
     */
    std::unique_lock<std::mutex> lock_history() {
        if (history_mutex_) {
            return std::unique_lock<std::mutex>(*history_mutex_);
        }
        return {};
    }

    void rollback_last_message() {
        auto lock = lock_history();
        history_->remove_last_message();
    }

    std::shared_ptr<backend::IBackend> backend_;
    std::shared_ptr<HistoryManager> history_;
    Config config_;
    std::mutex* history_mutex_;  // Optional mutex for thread-safe history access
    std::atomic<bool> cancelled_;
};

} // namespace engine
} // namespace zoo
