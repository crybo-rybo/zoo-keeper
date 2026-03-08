/**
 * @file agent.hpp
 * @brief Asynchronous orchestration layer that coordinates model inference and tool execution.
 */

#pragma once

#include "core/types.hpp"
#include "core/model.hpp"
#include "tools/registry.hpp"
#include "tools/parser.hpp"
#include "tools/validation.hpp"
#include "tools/interceptor.hpp"
#include "tools/grammar.hpp"
#include "internal/log.hpp"
#include <thread>
#include <future>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <unordered_map>
#include <chrono>

namespace zoo {

/**
 * @brief Internal request envelope processed by the agent worker thread.
 */
struct Request {
    Message message; ///< User message to append before processing begins.
    std::optional<std::function<void(std::string_view)>> streaming_callback; ///< Optional per-request streaming callback.
    std::chrono::steady_clock::time_point submitted_at; ///< Submission timestamp used for diagnostics and metrics.
    std::shared_ptr<std::promise<Expected<Response>>> promise; ///< Promise fulfilled when processing completes.
    RequestId id = 0; ///< Unique request identifier assigned by `Agent`.
    std::shared_ptr<std::atomic<bool>> cancelled; ///< Shared cancellation flag observed by the worker thread.

    /**
     * @brief Creates a request envelope and initializes its cancellation token.
     *
     * @param msg Message to process.
     * @param callback Optional callback that receives streamed text fragments.
     */
    Request(Message msg,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt)
        : message(std::move(msg))
        , streaming_callback(std::move(callback))
        , submitted_at(std::chrono::steady_clock::now())
        , cancelled(std::make_shared<std::atomic<bool>>(false))
    {}
};

/**
 * @brief Thread-safe queue used to hand requests to the inference thread.
 */
class RequestQueue {
public:
    /**
     * @brief Creates a queue with an optional maximum capacity.
     *
     * @param max_size Maximum number of queued requests, or `0` for unbounded.
     */
    explicit RequestQueue(size_t max_size = 0)
        : max_size_(max_size), shutdown_(false) {}

    /**
     * @brief Attempts to enqueue a request.
     *
     * @param request Request to enqueue.
     * @return `true` when the request was queued, or `false` if the queue is
     *         shutting down or already at capacity.
     */
    bool push(Request request) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutdown_) return false;
        if (max_size_ > 0 && queue_.size() >= max_size_) return false;
        queue_.push(std::move(request));
        cv_.notify_one();
        return true;
    }

    /**
     * @brief Waits for and pops the next queued request.
     *
     * @return The next request, or `std::nullopt` once shutdown has been
     *         signaled and the queue has been drained.
     */
    std::optional<Request> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) return std::nullopt;
        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

    /// Prevents new pushes and wakes any threads blocked in `pop()`.
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

private:
    std::queue<Request> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    bool shutdown_;
};

/**
 * @brief Handle returned by `Agent::chat()` for request tracking and result retrieval.
 */
    struct RequestHandle {
    RequestId id; ///< Request identifier accepted by the agent.
    std::future<Expected<Response>> future; ///< Future resolved with the final response or error.

    /// Creates an empty handle with an invalid request id.
    RequestHandle() noexcept : id(0) {}
    /**
     * @brief Creates a handle bound to a live request future.
     *
     * @param id Assigned request identifier.
     * @param future Future that resolves when processing completes.
     */
    RequestHandle(RequestId id, std::future<Expected<Response>> future)
        : id(id), future(std::move(future)) {}
    /// Moves ownership of the result future.
    RequestHandle(RequestHandle&&) noexcept = default;
    /// Moves ownership of the result future.
    RequestHandle& operator=(RequestHandle&&) noexcept = default;
    /// Request handles are non-copyable because `std::future` is move-only.
    RequestHandle(const RequestHandle&) = delete;
    /// Request handles are non-copyable because `std::future` is move-only.
    RequestHandle& operator=(const RequestHandle&) = delete;
};

/**
 * @brief Async orchestration layer built on top of `zoo::core::Model`.
 *
 * `Agent` owns a background inference thread, a request queue, and a tool
 * registry. It implements the tool loop of detect, validate, execute, inject,
 * and re-generate until the assistant produces a final user-visible response.
 */
class Agent {
public:
    /**
     * @brief Creates and starts an agent from the supplied configuration.
     *
     * @param config Runtime configuration used to load the underlying model.
     * @return A running agent, or an error if model creation fails.
     */
    static Expected<std::unique_ptr<Agent>> create(const Config& config) {
        auto model_result = core::Model::load(config);
        if (!model_result) {
            return std::unexpected(model_result.error());
        }

        return std::unique_ptr<Agent>(new Agent(config, std::move(*model_result)));
    }

    /// Stops the worker thread and releases owned resources.
    ~Agent() {
        stop();
    }

    /// Agents own background state and cannot be copied.
    Agent(const Agent&) = delete;
    /// Agents own background state and cannot be copied.
    Agent& operator=(const Agent&) = delete;
    /// Agents own thread-affine state and cannot be moved.
    Agent(Agent&&) = delete;
    /// Agents own thread-affine state and cannot be moved.
    Agent& operator=(Agent&&) = delete;

    /**
     * @brief Queues a user message for asynchronous processing.
     *
     * @param message User message to append to the conversation.
     * @param callback Optional callback that receives streamed visible text.
     * @return Handle containing the request id and result future. If the agent
     *         is not running or the queue rejects the request, the future is
     *         resolved immediately with an error.
     */
    RequestHandle chat(
        Message message,
        std::optional<std::function<void(std::string_view)>> callback = std::nullopt
    ) {
        auto promise = std::make_shared<std::promise<Expected<Response>>>();
        std::future<Expected<Response>> future = promise->get_future();
        RequestId request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);

        if (!running_.load(std::memory_order_acquire)) {
            promise->set_value(std::unexpected(Error{
                ErrorCode::AgentNotRunning,
                "Agent is not running"
            }));
            return RequestHandle{request_id, std::move(future)};
        }

        Request request(std::move(message), std::move(callback));
        request.promise = promise;
        request.id = request_id;

        {
            std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
            cancel_tokens_[request_id] = request.cancelled;
        }

        if (!request_queue_.push(std::move(request))) {
            promise->set_value(std::unexpected(Error{
                ErrorCode::QueueFull,
                "Request queue is full or agent is shutting down"
            }));
            std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
            cancel_tokens_.erase(request_id);
            return RequestHandle{request_id, std::move(future)};
        }

        return RequestHandle{request_id, std::move(future)};
    }

    /**
     * @brief Requests cancellation of a queued or running request.
     *
     * Cancellation is cooperative. Requests that have already completed or have
     * been cleaned up are unaffected.
     *
     * @param id Request identifier returned by `chat()`.
     */
    void cancel(RequestId id) {
        std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
        auto it = cancel_tokens_.find(id);
        if (it != cancel_tokens_.end()) {
            it->second->store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Replaces the current system prompt on the underlying model.
     *
     * @param prompt New system prompt content.
     */
    void set_system_prompt(const std::string& prompt) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        model_->set_system_prompt(prompt);
    }

    /// Stops the worker thread and prevents additional requests from being processed.
    void stop() {
        if (!running_.load(std::memory_order_acquire)) return;
        running_.store(false, std::memory_order_release);
        request_queue_.shutdown();
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }

    /// Returns whether the background inference thread is still accepting work.
    bool is_running() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /// Returns the immutable configuration used to create the agent.
    const Config& get_config() const noexcept { return config_; }

    /// Returns a snapshot of the underlying model conversation history.
    std::vector<Message> get_history() const {
        std::lock_guard<std::mutex> lock(model_mutex_);
        return model_->get_history();
    }

    /// Clears the underlying model conversation history.
    void clear_history() {
        std::lock_guard<std::mutex> lock(model_mutex_);
        model_->clear_history();
    }

    /**
     * @brief Registers a strongly typed tool and refreshes grammar constraints.
     *
     * @tparam Func Callable type to register.
     * @param name Public tool name.
     * @param description Human-readable description for prompts and schemas.
     * @param param_names Parameter names in callable argument order.
     * @param func Callable implementation.
     * @return Empty success when registered, or the underlying registry error.
     */
    template<typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                       const std::vector<std::string>& param_names, Func func) {
        auto result = tool_registry_.register_tool(name, description, param_names, std::move(func));
        if (result) {
            update_tool_grammar();
        }
        return result;
    }

    /// Returns the number of tools currently registered with the agent.
    size_t tool_count() const noexcept {
        return tool_registry_.size();
    }

    /**
     * @brief Builds a system prompt that advertises the currently registered tools.
     *
     * When grammar-based tool calling is active the prompt describes the
     * sentinel-tagged format; otherwise it falls back to plain JSON instructions.
     *
     * @param base_prompt Base system prompt to extend.
     * @return Prompt text augmented with tool usage instructions and schemas.
     */
    std::string build_tool_system_prompt(const std::string& base_prompt) const {
        auto schemas = tool_registry_.get_all_schemas();
        if (schemas.empty()) return base_prompt;

        if (model_->has_tool_grammar()) {
            // Sentinel-based instructions — grammar constrains the output format
            return base_prompt
                + "\n\nYou have access to tools. When you need to use a tool, wrap the "
                  "call in sentinel tags like this:\n"
                  "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}</tool_call>\n"
                  "\nYou may think step-by-step before calling a tool. "
                  "Do NOT output any text after the </tool_call> closing tag.\n"
                  "After receiving a tool result, incorporate it into a natural response.\n"
                  "If no tool is needed, respond normally without sentinel tags.\n"
                  "\nAvailable tools:\n"
                + schemas.dump(2);
        }

        // Heuristic fallback instructions
        return base_prompt
            + "\n\nWhen you need to use a tool, respond with a JSON object containing "
              "\"name\" and \"arguments\" fields. For example:\n"
              "{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n"
              "\nOutput ONLY the JSON tool call when invoking a tool — no text after it.\n"
              "After receiving a tool result, incorporate it into a natural response.\n"
              "If no tool is needed, respond normally without JSON.\n"
              "\nAvailable tools:\n"
            + schemas.dump(2);
    }

private:
    /**
     * @brief Constructs a running agent around an initialized model.
     *
     * @param config Immutable runtime configuration.
     * @param model Initialized model instance owned by the agent.
     */
    Agent(const Config& config, std::unique_ptr<core::Model> model)
        : config_(config)
        , model_(std::move(model))
        , request_queue_(config.request_queue_capacity)
        , running_(true)
    {
        inference_thread_ = std::thread([this]() { inference_loop(); });
    }

    /// Main worker loop that drains queued requests until the agent stops.
    void inference_loop() {
        try {
            while (running_.load(std::memory_order_acquire)) {
                auto request_opt = request_queue_.pop();
                if (!request_opt) break;

                auto promise = request_opt->promise;

                // Check per-request cancellation
                if (request_opt->cancelled &&
                    request_opt->cancelled->load(std::memory_order_acquire)) {
                    ZOO_LOG("info", "request %lu cancelled before processing", (unsigned long)request_opt->id);
                    if (promise) {
                        promise->set_value(std::unexpected(Error{
                            ErrorCode::RequestCancelled, "Request cancelled"
                        }));
                    }
                    cleanup_cancel_token(request_opt->id);
                    continue;
                }

                Expected<Response> result;
                try {
                    result = process_request(*request_opt);
                } catch (const std::exception& e) {
                    ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
                    result = std::unexpected(Error{
                        ErrorCode::InferenceFailed,
                        std::string("Unhandled exception: ") + e.what()
                    });
                } catch (...) {
                    ZOO_LOG("error", "unknown exception in inference thread");
                    result = std::unexpected(Error{
                        ErrorCode::InferenceFailed,
                        "Unknown exception in inference thread"
                    });
                }

                cleanup_cancel_token(request_opt->id);

                if (promise) {
                    promise->set_value(std::move(result));
                }
            }

            fail_pending_requests(Error{
                ErrorCode::AgentNotRunning,
                "Agent stopped before request could be processed"
            });
        } catch (const std::exception& e) {
            ZOO_LOG("error", "fatal exception escaped inference thread: %s", e.what());
            fail_pending_requests(Error{
                ErrorCode::InferenceFailed,
                std::string("Inference thread terminated unexpectedly: ") + e.what()
            });
        } catch (...) {
            ZOO_LOG("error", "fatal unknown exception escaped inference thread");
            fail_pending_requests(Error{
                ErrorCode::InferenceFailed,
                "Inference thread terminated unexpectedly"
            });
        }
    }

    /**
     * @brief Processes one request, including any tool-call iterations.
     *
     * @param request Request envelope to process.
     * @return Final assistant response or an error encountered during processing.
     */
    Expected<Response> process_request(const Request& request) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        auto start_time = std::chrono::steady_clock::now();

        // Add user message
        auto add_result = model_->add_message(request.message);
        if (!add_result) {
            return std::unexpected(add_result.error());
        }

        // Metrics tracking
        std::chrono::steady_clock::time_point first_token_time;
        bool first_token_received = false;
        int total_completion_tokens = 0;
        int total_prompt_tokens = 0;
        std::vector<Message> tool_call_history;

        tools::ErrorRecovery error_recovery(config_.max_tool_retries);
        int iteration = 0;
        const int max_tool_iterations = config_.max_tool_iterations;
        const bool has_tools = tool_registry_.size() > 0;
        const bool use_grammar_path = has_tools && model_->has_tool_grammar();

        ZOO_LOG("debug", "processing request %lu (tools=%d, grammar=%d)",
            (unsigned long)request.id, has_tools, use_grammar_path);

        while (iteration < max_tool_iterations) {
            ++iteration;

            // Check cancellation
            if (request.cancelled && request.cancelled->load(std::memory_order_acquire)) {
                return std::unexpected(Error{
                    ErrorCode::RequestCancelled,
                    "Request cancelled during tool loop"
                });
            }

            int completion_tokens = 0;

            // Metrics-tracking wrapper that forwards to an inner callback
            auto make_metrics_callback = [&](auto inner_callback) -> TokenCallback {
                return [&, cb = std::move(inner_callback)](std::string_view token) -> TokenAction {
                    if (!first_token_received) {
                        first_token_time = std::chrono::steady_clock::now();
                        first_token_received = true;
                    }
                    ++completion_tokens;
                    return cb(token);
                };
            };

            // Build the appropriate callback chain
            std::optional<TokenCallback> callback;
            std::optional<tools::ToolCallInterceptor> interceptor;

            if (use_grammar_path) {
                // Grammar path: stream directly to user callback.
                // Model handles sentinel detection internally — it stops streaming
                // tokens to the callback once <tool_call> appears, so code blocks
                // with braces flow through without any buffering or freezing.
                if (request.streaming_callback) {
                    callback = make_metrics_callback(
                        [&](std::string_view token) -> TokenAction {
                            (*request.streaming_callback)(token);
                            return TokenAction::Continue;
                        }
                    );
                } else {
                    callback = make_metrics_callback(
                        [](std::string_view) -> TokenAction { return TokenAction::Continue; }
                    );
                }
            } else if (has_tools) {
                // Heuristic fallback: use ToolCallInterceptor (brace-based buffering)
                interceptor.emplace(request.streaming_callback);
                auto interceptor_cb = interceptor->make_callback();
                callback = make_metrics_callback(std::move(interceptor_cb));
            } else if (request.streaming_callback) {
                // No tools — stream directly
                callback = make_metrics_callback(
                    [&](std::string_view token) -> TokenAction {
                        (*request.streaming_callback)(token);
                        return TokenAction::Continue;
                    }
                );
            } else {
                callback = make_metrics_callback(
                    [](std::string_view) -> TokenAction { return TokenAction::Continue; }
                );
            }

            auto generated = model_->generate_from_history(
                std::move(callback),
                [&request]() {
                    return request.cancelled &&
                        request.cancelled->load(std::memory_order_acquire);
                }
            );

            if (!generated) {
                return std::unexpected(generated.error());
            }

            total_completion_tokens += completion_tokens;
            total_prompt_tokens += generated->prompt_tokens;

            // Determine if a tool call was detected
            std::optional<tools::ToolCall> detected_tool_call;
            std::string response_text;

            if (use_grammar_path && generated->tool_call_detected) {
                // Grammar path: Model already detected the sentinel and stopped.
                // Parse the full output to extract chain-of-thought text and tool call.
                auto sentinel_result = tools::ToolCallParser::parse_sentinel(generated->text);
                detected_tool_call = std::move(sentinel_result.tool_call);
                response_text = std::move(sentinel_result.text_before);
            } else if (interceptor) {
                // Heuristic path: interceptor buffered and parsed
                auto intercept_result = interceptor->finalize();
                detected_tool_call = std::move(intercept_result.tool_call);
                response_text = std::move(intercept_result.full_text);
            } else {
                response_text = std::move(generated->text);
            }

            if (detected_tool_call.has_value()) {
                const auto& tc = *detected_tool_call;

                // Commit assistant message (chain-of-thought + tool call)
                model_->add_message(Message::assistant(
                    use_grammar_path ? generated->text : response_text));
                model_->finalize_response();

                if (!use_grammar_path) {
                    // Heuristic path: validate arguments (grammar already constrains these)
                    auto validation_error = error_recovery.validate_args(tc, tool_registry_);
                    if (!validation_error.empty()) {
                        if (!error_recovery.can_retry(tc.name)) {
                            ZOO_LOG("error", "tool retries exhausted for '%s': %s",
                                tc.name.c_str(), validation_error.c_str());
                            return std::unexpected(Error{
                                ErrorCode::ToolRetriesExhausted,
                                "Tool retries exhausted for '" + tc.name + "': " + validation_error
                            });
                        }
                        error_recovery.record_retry(tc.name);
                        ZOO_LOG("warn", "tool '%s' validation failed (retry %d/%d): %s",
                            tc.name.c_str(), error_recovery.get_retry_count(tc.name),
                            config_.max_tool_retries, validation_error.c_str());

                        std::string error_content = "Error: " + validation_error;
                        model_->add_message(Message::tool(
                            error_content + "\nPlease correct the arguments.", tc.id));
                        tool_call_history.push_back(
                            Message::tool(std::move(error_content), tc.id));
                        continue;
                    }
                }

                // Execute tool
                ZOO_LOG("info", "invoking tool '%s' (iteration %d, grammar=%d)",
                    tc.name.c_str(), iteration, use_grammar_path);
                auto invoke_result = tool_registry_.invoke(tc.name, tc.arguments);
                std::string tool_result_str;
                if (invoke_result) {
                    tool_result_str = invoke_result->dump();
                } else {
                    tool_result_str = "Error: " + invoke_result.error().message;
                }

                Message tool_msg = Message::tool(std::move(tool_result_str), tc.id);
                model_->add_message(tool_msg);
                tool_call_history.push_back(std::move(tool_msg));
                continue;
            }

            // If the model emitted EOG immediately after a tool result,
            // nudge it to produce a natural language response for the user.
            if (response_text.empty() && !tool_call_history.empty() &&
                iteration < max_tool_iterations) {
                model_->add_message(Message::user(
                    "Please respond to the user with the tool result."));
                continue;
            }

            // No tool call — final response
            auto end_time = std::chrono::steady_clock::now();

            // Commit assistant response
            model_->add_message(Message::assistant(response_text));
            model_->finalize_response();

            Response response;
            response.text = std::move(response_text);
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

        ZOO_LOG("error", "tool loop iteration limit reached (%d)", max_tool_iterations);
        return std::unexpected(Error{
            ErrorCode::ToolLoopLimitReached,
            "Tool loop iteration limit reached (" +
                std::to_string(max_tool_iterations) + ")"
        });
    }

    /**
     * @brief Removes a request's cancellation token after processing completes.
     *
     * @param id Request identifier to clean up.
     */
    void cleanup_cancel_token(RequestId id) {
        std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
        cancel_tokens_.erase(id);
    }

    /// Resolves every queued request with the supplied terminal error.
    void fail_pending_requests(const Error& error) {
        running_.store(false, std::memory_order_release);
        request_queue_.shutdown();

        while (auto remaining = request_queue_.pop()) {
            if (remaining->promise) {
                remaining->promise->set_value(std::unexpected(error));
            }
            cleanup_cancel_token(remaining->id);
        }
    }

    /// Rebuilds the model's tool grammar from the currently registered tool schemas.
    void update_tool_grammar() {
        std::lock_guard<std::mutex> lock(model_mutex_);
        auto schemas = tool_registry_.get_all_schemas();
        auto grammar = tools::GrammarBuilder::build(schemas);
        if (grammar.empty()) {
            model_->clear_tool_grammar();
        } else if (model_->set_tool_grammar(grammar)) {
            ZOO_LOG("info", "tool grammar updated (%zu tools)", tool_registry_.size());
        } else {
            ZOO_LOG("warn", "grammar sampler init failed, falling back to unconstrained generation");
        }
    }

    Config config_;
    std::unique_ptr<core::Model> model_;
    mutable std::mutex model_mutex_;
    tools::ToolRegistry tool_registry_;
    RequestQueue request_queue_;

    std::thread inference_thread_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> next_request_id_{1};

    std::mutex cancel_tokens_mutex_;
    std::unordered_map<RequestId, std::shared_ptr<std::atomic<bool>>> cancel_tokens_;
};

} // namespace zoo
