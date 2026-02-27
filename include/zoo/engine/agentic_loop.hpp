#pragma once

#include "../types.hpp"
#include "../backend/IBackend.hpp"
#include "history_manager.hpp"
#include "request_queue.hpp"
#include "tool_registry.hpp"
#include "tool_call_parser.hpp"
#include "error_recovery.hpp"
#include "rag_store.hpp"
#include "context_database.hpp"
#include <atomic>
#include <algorithm>
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
 * - Detects tool calls in model output
 * - Validates args, executes tools, injects results, and loops
 * - Adds assistant responses to HistoryManager
 * - Calls backend->finalize_response() to update prompt cache
 * - Tracks metrics (latency, TTFT, tokens/sec)
 * - Supports cancellation via atomic flag
 */
class AgenticLoop {
public:
    AgenticLoop(
        std::shared_ptr<backend::IBackend> backend,
        std::shared_ptr<HistoryManager> history,
        const Config& config
    )
        : backend_(std::move(backend))
        , history_(std::move(history))
        , config_(config)
        , cancelled_(false)
    {}

    /**
     * @brief Set the tool registry for tool calling support
     */
    void set_tool_registry(std::shared_ptr<ToolRegistry> registry) {
        tool_registry_ = std::move(registry);
    }

    /**
     * @brief Set the retriever used for RAG queries.
     */
    void set_retriever(std::shared_ptr<IRetriever> retriever) {
        retriever_ = std::move(retriever);
    }

    /**
     * @brief Enable durable context database memory for automatic long-context retrieval.
     */
    void set_context_database(std::shared_ptr<ContextDatabase> context_database) {
        context_database_ = std::move(context_database);
        if (context_database_) {
            retriever_ = context_database_;
        }
    }

    /**
     * @brief Set max tool loop iterations (default: 5)
     */
    void set_max_tool_iterations(int max) {
        max_tool_iterations_ = max;
    }

    /**
     * @brief Set minimum token headroom reserved for the response (default: 256)
     *
     * If remaining context after the prompt is less than this value, the
     * request fails with ContextWindowExceeded before generation begins.
     */
    void set_min_response_reserve(int reserve) {
        min_response_reserve_ = reserve;
    }

    /**
     * @brief Process a single request (with optional tool loop)
     *
     * @param request The request to process
     * @param per_request_cancel Optional per-request cancellation token.
     *        When set to true, the request will be cancelled at the next
     *        iteration boundary (tool loop check or before processing).
     */
    Expected<Response> process_request(
        const Request& request,
        std::shared_ptr<std::atomic<bool>> per_request_cancel = nullptr
    ) {
        bool using_ephemeral_rag = false;
        auto finish = [this, &using_ephemeral_rag](Expected<Response> result) -> Expected<Response> {
            if (using_ephemeral_rag) {
                // Ephemeral prompt injection changes template/KV state relative to persisted history.
                // Reset backend state to keep subsequent turns consistent.
                backend_->clear_kv_cache();
            }
            return result;
        };

        // Helper to check both global and per-request cancellation
        auto is_cancelled = [this, &per_request_cancel]() {
            if (cancelled_.load(std::memory_order_acquire)) return true;
            if (per_request_cancel && per_request_cancel->load(std::memory_order_acquire)) return true;
            return false;
        };

        // Check cancellation
        if (is_cancelled()) {
            return finish(tl::unexpected(Error{
                ErrorCode::RequestCancelled,
                "Request cancelled"
            }));
        }

        auto start_time = std::chrono::steady_clock::now();
        size_t user_message_index = 0;

        // Add user message to history
        std::vector<Message> pruned_messages;
        {
            if (auto result = history_->add_message(request.message); !result) {
                return finish(tl::unexpected(result.error()));
            }
            user_message_index = history_->get_messages().size() - 1;

            if (history_->is_context_exceeded()) {
                if (context_database_) {
                    const int target_tokens = std::max(
                        1,
                        static_cast<int>(
                            static_cast<double>(history_->get_context_size()) * memory_prune_target_ratio_));
                    pruned_messages = history_->prune_oldest_messages_until(
                        target_tokens,
                        memory_min_messages_to_keep_);
                    user_message_index = history_->get_messages().size() - 1;
                }

                if (history_->is_context_exceeded()) {
                    history_->remove_last_message();
                    return finish(tl::unexpected(Error{
                        ErrorCode::ContextWindowExceeded,
                        "Context window exceeded",
                        "Estimated tokens: " + std::to_string(history_->get_estimated_tokens()) +
                        " / " + std::to_string(history_->get_context_size())
                    }));
                }
            }
        }

        if (!pruned_messages.empty() && context_database_) {
            auto archive_result = context_database_->add_messages(pruned_messages, "history_archive");
            if (!archive_result) {
                history_->prepend_messages(pruned_messages);
                history_->remove_last_message();
                return finish(tl::unexpected(archive_result.error()));
            }
        }

        std::vector<RagChunk> rag_chunks;
        std::string rag_context;
        const bool use_retrieval = request.options.rag.enabled || static_cast<bool>(context_database_);
        if (use_retrieval) {
            if (request.options.rag.context_override.has_value()) {
                rag_context = *request.options.rag.context_override;
            } else if (retriever_) {
                const int retrieval_top_k =
                    request.options.rag.enabled ? request.options.rag.top_k : memory_retrieval_top_k_;
                auto rag_result = retriever_->retrieve(RagQuery{
                    request.message.content,
                    retrieval_top_k
                });
                if (!rag_result) {
                    rollback_last_message();
                    return finish(tl::unexpected(rag_result.error()));
                }
                rag_chunks = std::move(*rag_result);
                rag_context = format_rag_context(rag_chunks);
            }
        }

        using_ephemeral_rag = !rag_context.empty();
        if (using_ephemeral_rag) {
            // Force a clean backend state before this turn because the prompt contains
            // ephemeral context that is intentionally not stored in HistoryManager.
            backend_->clear_kv_cache();
        }

        // Metrics tracking across tool loop iterations
        std::chrono::steady_clock::time_point first_token_time;
        bool first_token_received = false;
        int total_completion_tokens = 0;
        int total_prompt_tokens = 0;
        std::vector<Message> tool_call_history;
        tool_call_history.reserve(max_tool_iterations_);

        ErrorRecovery error_recovery;
        int iteration = 0;

        while (iteration < max_tool_iterations_) {
            ++iteration;

            // Check cancellation (global and per-request)
            if (is_cancelled()) {
                return finish(tl::unexpected(Error{
                    ErrorCode::RequestCancelled,
                    "Request cancelled during tool loop"
                }));
            }

            // Format prompt — avoid copying history when there is no RAG context
            std::string prompt;
            {
                Expected<std::string> prompt_result;
                if (rag_context.empty()) {
                    prompt_result = backend_->format_prompt(history_->get_messages());
                } else {
                    auto prompt_messages = build_prompt_messages(
                        history_->get_messages(),
                        user_message_index,
                        rag_context
                    );
                    prompt_result = backend_->format_prompt(prompt_messages);
                }
                if (!prompt_result) {
                    rollback_last_message();
                    return finish(tl::unexpected(prompt_result.error()));
                }
                prompt = std::move(*prompt_result);
            }

            // Tokenize
            auto tokens_result = backend_->tokenize(prompt);
            if (!tokens_result) {
                rollback_last_message();
                return finish(tl::unexpected(tokens_result.error()));
            }
            const auto& prompt_tokens = *tokens_result;
            total_prompt_tokens += static_cast<int>(prompt_tokens.size());

            // Pre-generate safety check: if the formatted prompt alone exceeds the
            // context size, generation will certainly fail. Catch it here with a
            // descriptive error instead of letting it fail inside generate().
            {
                int prompt_size = static_cast<int>(prompt_tokens.size());
                int ctx_size = backend_->get_context_size();

                if (prompt_size > ctx_size) {
                    rollback_last_message();
                    return finish(tl::unexpected(Error{
                        ErrorCode::ContextWindowExceeded,
                        "Context window exceeded: formatted prompt requires " +
                        std::to_string(prompt_size) +
                        " tokens but context size is " + std::to_string(ctx_size) +
                        " (chat template overhead not captured by token estimator)"
                    }));
                }

                // Headroom check: ensure there is enough room in the context for a
                // meaningful response. cap min_response_reserve_ at remaining ctx space
                // when max_tokens is set (avoids requiring more than the user asked for).
                const int remaining_ctx = ctx_size - prompt_size;
                const int effective_reserve = (config_.max_tokens > 0)
                    ? std::min(min_response_reserve_, config_.max_tokens)
                    : min_response_reserve_;
                if (remaining_ctx < effective_reserve) {
                    rollback_last_message();
                    return finish(tl::unexpected(Error{
                        ErrorCode::ContextWindowExceeded,
                        "Insufficient context headroom for response: " +
                        std::to_string(remaining_ctx) + " tokens remaining, " +
                        std::to_string(effective_reserve) + " required"
                    }));
                }
            }

            // Generate with streaming callback
            int completion_tokens = 0;
            auto wrapped_callback = make_streaming_callback(
                request, first_token_time, first_token_received, completion_tokens);

            auto generate_result = backend_->generate(
                prompt_tokens,
                config_.max_tokens,
                config_.stop_sequences,
                wrapped_callback
            );

            if (!generate_result) {
                rollback_last_message();
                return finish(tl::unexpected(generate_result.error()));
            }

            total_completion_tokens += completion_tokens;
            // Take ownership to enable moves (avoids copying LLM output)
            std::string generated_text = std::move(*generate_result);

            // Check for tool calls (only if registry is available and has tools)
            if (tool_registry_ && tool_registry_->size() > 0) {
                auto parse_result = ToolCallParser::parse(generated_text);

                if (parse_result.tool_call.has_value()) {
                    const auto& tc = *parse_result.tool_call;

                    // Add assistant message with tool call to history
                    if (auto err = commit_assistant_response(generated_text); !err) {
                        return finish(tl::unexpected(err.error()));
                    }

                    // Validate arguments
                    auto validation_error = error_recovery.validate_args(tc, *tool_registry_);
                    if (!validation_error.empty()) {
                        // Validation failed
                        if (!error_recovery.can_retry(tc.name)) {
                            return finish(tl::unexpected(Error{
                                ErrorCode::ToolRetriesExhausted,
                                "Tool retries exhausted for '" + tc.name + "': " + validation_error
                            }));
                        }

                        error_recovery.record_retry(tc.name);

                        // Build error content once, reuse for history and tracking
                        std::string error_content = "Error: " + validation_error;
                        (void)history_->add_message(Message::tool(
                            error_content + "\nPlease correct the arguments.", tc.id));
                        tool_call_history.push_back(
                            Message::tool(std::move(error_content), tc.id));
                        continue;  // Loop back for retry
                    }

                    // Execute tool
                    auto invoke_result = tool_registry_->invoke(tc.name, tc.arguments);
                    std::string tool_result_str;
                    if (invoke_result) {
                        tool_result_str = invoke_result->dump();
                    } else {
                        tool_result_str = "Error: " + invoke_result.error().message;
                    }

                    // Build tool result message once; copy to history, move to tracking
                    Message tool_msg = Message::tool(std::move(tool_result_str), tc.id);
                    auto add_result = history_->add_message(tool_msg);
                    if (!add_result) {
                        return finish(tl::unexpected(add_result.error()));
                    }
                    tool_call_history.push_back(std::move(tool_msg));

                    continue;  // Loop back for model to process tool result
                }
            }

            // No tool call detected — this is the final response
            auto end_time = std::chrono::steady_clock::now();

            // Add assistant response to history
            if (auto err = commit_assistant_response(generated_text); !err) {
                return finish(tl::unexpected(err.error()));
            }

            // Build response
            Response response;
            response.text = std::move(generated_text);
            response.tool_calls = std::move(tool_call_history);
            response.rag_chunks = std::move(rag_chunks);

            response.usage.prompt_tokens = total_prompt_tokens;
            response.usage.completion_tokens = total_completion_tokens;
            response.usage.total_tokens = total_prompt_tokens + total_completion_tokens;

            auto total_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
            response.metrics.latency_ms = total_latency;

            if (first_token_received) {
                response.metrics.time_to_first_token_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        first_token_time - start_time);

                auto generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - first_token_time);
                if (generation_time.count() > 0) {
                    response.metrics.tokens_per_second =
                        (total_completion_tokens * 1000.0) / generation_time.count();
                }
            }

            return finish(response);
        }

        // If we get here, we've exceeded the tool loop limit
        return finish(tl::unexpected(Error{
            ErrorCode::ToolLoopLimitReached,
            "Tool loop iteration limit reached (" +
                std::to_string(max_tool_iterations_) + ")"
        }));
    }

    /**
     * @brief Run the main inference loop (legacy helper)
     */
    void run(RequestQueue& queue) {
        while (!cancelled_.load(std::memory_order_acquire)) {
            auto request_opt = queue.pop();
            if (!request_opt) break;
            auto result = process_request(*request_opt);
            (void)result;
        }
    }

    void cancel() {
        cancelled_.store(true, std::memory_order_release);
    }

    bool is_cancelled() const {
        return cancelled_.load(std::memory_order_acquire);
    }

    void reset() {
        cancelled_.store(false, std::memory_order_release);
    }

private:
    /** @brief Remove the last message from history (error recovery). */
    void rollback_last_message() {
        history_->remove_last_message();
    }

    /** @brief Add assistant message to history and finalize backend state. */
    Expected<void> commit_assistant_response(const std::string& text) {
        auto add_result = history_->add_message(Message::assistant(text));
        if (!add_result) {
            return tl::unexpected(add_result.error());
        }
        backend_->finalize_response(history_->get_messages());
        return {};
    }

    /** @brief Create a streaming callback that tracks metrics and forwards to user callback. */
    std::optional<std::function<void(std::string_view)>> make_streaming_callback(
        const Request& request,
        std::chrono::steady_clock::time_point& first_token_time,
        bool& first_token_received,
        int& completion_tokens
    ) {
        if (request.streaming_callback) {
            return [&, user_callback = *request.streaming_callback](std::string_view token) {
                if (!first_token_received) {
                    first_token_time = std::chrono::steady_clock::now();
                    first_token_received = true;
                }
                ++completion_tokens;
                user_callback(token);
            };
        } else {
            return [&](std::string_view) {
                if (!first_token_received) {
                    first_token_time = std::chrono::steady_clock::now();
                    first_token_received = true;
                }
                ++completion_tokens;
            };
        }
    }

    std::shared_ptr<backend::IBackend> backend_;
    std::shared_ptr<HistoryManager> history_;
    Config config_;
    std::atomic<bool> cancelled_;
    std::shared_ptr<ToolRegistry> tool_registry_;
    std::shared_ptr<IRetriever> retriever_;
    std::shared_ptr<ContextDatabase> context_database_;
    int max_tool_iterations_ = 5;
    int min_response_reserve_ = 256;
    int memory_retrieval_top_k_ = 4;
    int memory_min_messages_to_keep_ = 6;
    double memory_prune_target_ratio_ = 0.75;

    std::vector<Message> build_prompt_messages(
        const std::vector<Message>& history_messages,
        size_t user_message_index,
        const std::string& rag_context
    ) const {
        if (rag_context.empty()) {
            return history_messages;
        }

        std::vector<Message> prompt_messages;
        prompt_messages.reserve(history_messages.size() + 1);

        const size_t insert_at = std::min(user_message_index, history_messages.size());
        prompt_messages.insert(
            prompt_messages.end(),
            history_messages.begin(),
            history_messages.begin() + static_cast<std::vector<Message>::difference_type>(insert_at));

        // Ephemeral system message so the model treats retrieved text as grounding context.
        prompt_messages.push_back(Message::system(rag_context));

        prompt_messages.insert(
            prompt_messages.end(),
            history_messages.begin() + static_cast<std::vector<Message>::difference_type>(insert_at),
            history_messages.end());

        return prompt_messages;
    }

    static std::string format_rag_context(const std::vector<RagChunk>& chunks) {
        if (chunks.empty()) {
            return {};
        }

        std::string out;
        out.reserve(512);
        out += "Use the following retrieved context when relevant.\n";
        out += "If the context is insufficient, say what is missing.\n";
        out += "\nRetrieved Context:\n";

        for (size_t i = 0; i < chunks.size(); ++i) {
            out += "[" + std::to_string(i + 1) + "] ";
            if (chunks[i].source.has_value()) {
                out += "source=" + *chunks[i].source + " ";
            }
            out += "chunk_id=" + chunks[i].id + "\n";
            out += chunks[i].content;
            out += "\n\n";
        }

        return out;
    }
};

} // namespace engine
} // namespace zoo
