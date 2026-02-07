#pragma once

#include "../types.hpp"
#include "../backend/IBackend.hpp"
#include "history_manager.hpp"
#include "request_queue.hpp"
#include "tool_registry.hpp"
#include "tool_call_parser.hpp"
#include "error_recovery.hpp"
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
     * @brief Set the tool registry for tool calling support
     */
    void set_tool_registry(std::shared_ptr<ToolRegistry> registry) {
        tool_registry_ = std::move(registry);
    }

    /**
     * @brief Set max tool loop iterations (default: 5)
     */
    void set_max_tool_iterations(int max) {
        max_tool_iterations_ = max;
    }

    /**
     * @brief Process a single request (with optional tool loop)
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

        // Add user message to history (with lock)
        {
            auto lock = lock_history();
            if (auto result = history_->add_message(request.message); !result) {
                return tl::unexpected(result.error());
            }

            if (history_->is_context_exceeded()) {
                history_->remove_last_message();
                return tl::unexpected(Error{
                    ErrorCode::ContextWindowExceeded,
                    "Context window exceeded",
                    "Estimated tokens: " + std::to_string(history_->get_estimated_tokens()) +
                    " / " + std::to_string(history_->get_context_size())
                });
            }
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

            // Check cancellation
            if (cancelled_.load(std::memory_order_acquire)) {
                return tl::unexpected(Error{
                    ErrorCode::RequestCancelled,
                    "Request cancelled during tool loop"
                });
            }

            // Format prompt
            std::string prompt;
            {
                auto lock = lock_history();
                auto prompt_result = backend_->format_prompt(history_->get_messages());
                if (!prompt_result) {
                    rollback_last_message();
                    return tl::unexpected(prompt_result.error());
                }
                prompt = std::move(*prompt_result);
            }

            // Tokenize
            auto tokens_result = backend_->tokenize(prompt);
            if (!tokens_result) {
                rollback_last_message();
                return tl::unexpected(tokens_result.error());
            }
            const auto& prompt_tokens = *tokens_result;
            total_prompt_tokens += static_cast<int>(prompt_tokens.size());

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
                return tl::unexpected(generate_result.error());
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
                        return tl::unexpected(err.error());
                    }

                    // Validate arguments
                    auto validation_error = error_recovery.validate_args(tc, *tool_registry_);
                    if (!validation_error.empty()) {
                        // Validation failed
                        if (!error_recovery.can_retry(tc.name)) {
                            return tl::unexpected(Error{
                                ErrorCode::ToolRetriesExhausted,
                                "Tool retries exhausted for '" + tc.name + "': " + validation_error
                            });
                        }

                        error_recovery.record_retry(tc.name);

                        // Build error content once, reuse for history and tracking
                        std::string error_content = "Error: " + validation_error;
                        {
                            auto lock = lock_history();
                            (void)history_->add_message(Message::tool(
                                error_content + "\nPlease correct the arguments.", tc.id));
                        }
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
                    {
                        auto lock = lock_history();
                        auto add_result = history_->add_message(tool_msg);
                        if (!add_result) {
                            return tl::unexpected(add_result.error());
                        }
                    }
                    tool_call_history.push_back(std::move(tool_msg));

                    continue;  // Loop back for model to process tool result
                }
            }

            // No tool call detected — this is the final response
            auto end_time = std::chrono::steady_clock::now();

            // Add assistant response to history
            if (auto err = commit_assistant_response(generated_text); !err) {
                return tl::unexpected(err.error());
            }

            // Build response
            Response response;
            response.text = std::move(generated_text);
            response.tool_calls = std::move(tool_call_history);

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

            return response;
        }

        // If we get here, we've exceeded the tool loop limit
        return tl::unexpected(Error{
            ErrorCode::ToolLoopLimitReached,
            "Tool loop iteration limit reached (" +
                std::to_string(max_tool_iterations_) + ")"
        });
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

    Expected<void> commit_assistant_response(const std::string& text) {
        auto lock = lock_history();
        auto add_result = history_->add_message(Message::assistant(text));
        if (!add_result) {
            return tl::unexpected(add_result.error());
        }
        backend_->finalize_response(history_->get_messages());
        return {};
    }

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
    std::mutex* history_mutex_;
    std::atomic<bool> cancelled_;
    std::shared_ptr<ToolRegistry> tool_registry_;
    int max_tool_iterations_ = 5;
};

} // namespace engine
} // namespace zoo
