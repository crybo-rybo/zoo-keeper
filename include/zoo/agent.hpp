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

// Internal request representation
struct Request {
    Message message;
    std::optional<std::function<void(std::string_view)>> streaming_callback;
    std::chrono::steady_clock::time_point submitted_at;
    std::shared_ptr<std::promise<Expected<Response>>> promise;
    RequestId id = 0;
    std::shared_ptr<std::atomic<bool>> cancelled;

    Request(Message msg,
            std::optional<std::function<void(std::string_view)>> callback = std::nullopt)
        : message(std::move(msg))
        , streaming_callback(std::move(callback))
        , submitted_at(std::chrono::steady_clock::now())
        , cancelled(std::make_shared<std::atomic<bool>>(false))
    {}
};

// Thread-safe request queue
class RequestQueue {
public:
    explicit RequestQueue(size_t max_size = 0)
        : max_size_(max_size), shutdown_(false) {}

    bool push(Request request) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutdown_) return false;
        if (max_size_ > 0 && queue_.size() >= max_size_) return false;
        queue_.push(std::move(request));
        cv_.notify_one();
        return true;
    }

    std::optional<Request> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) return std::nullopt;
        Request req = std::move(queue_.front());
        queue_.pop();
        return req;
    }

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

// Handle returned from Agent::chat()
struct RequestHandle {
    RequestId id;
    std::future<Expected<Response>> future;

    RequestHandle() : id(0) {}
    RequestHandle(RequestId id, std::future<Expected<Response>> future)
        : id(id), future(std::move(future)) {}
    RequestHandle(RequestHandle&&) = default;
    RequestHandle& operator=(RequestHandle&&) = default;
    RequestHandle(const RequestHandle&) = delete;
    RequestHandle& operator=(const RequestHandle&) = delete;
};

/**
 * Agent is the async orchestration layer.
 *
 * It composes a core::Model and tools::ToolRegistry, runs an inference thread,
 * and implements the agentic tool loop (detect -> validate -> execute -> inject -> re-generate).
 */
class Agent {
public:
    static Expected<std::unique_ptr<Agent>> create(const Config& config) {
        auto model_result = core::Model::load(config);
        if (!model_result) {
            return std::unexpected(model_result.error());
        }

        return std::unique_ptr<Agent>(new Agent(config, std::move(*model_result)));
    }

    ~Agent() {
        stop();
    }

    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;
    Agent(Agent&&) = delete;
    Agent& operator=(Agent&&) = delete;

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

    void cancel(RequestId id) {
        std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
        auto it = cancel_tokens_.find(id);
        if (it != cancel_tokens_.end()) {
            it->second->store(true, std::memory_order_release);
        }
    }

    void set_system_prompt(const std::string& prompt) {
        std::lock_guard<std::mutex> lock(model_mutex_);
        model_->set_system_prompt(prompt);
    }

    void stop() {
        if (!running_.load(std::memory_order_acquire)) return;
        running_.store(false, std::memory_order_release);
        request_queue_.shutdown();
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
    }

    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

    const Config& get_config() const { return config_; }

    std::vector<Message> get_history() const {
        std::lock_guard<std::mutex> lock(model_mutex_);
        return model_->get_history();
    }

    void clear_history() {
        std::lock_guard<std::mutex> lock(model_mutex_);
        model_->clear_history();
    }

    template<typename Func>
    Expected<void> register_tool(const std::string& name, const std::string& description,
                       const std::vector<std::string>& param_names, Func func) {
        auto result = tool_registry_.register_tool(name, description, param_names, std::move(func));
        if (result) {
            update_tool_grammar();
        }
        return result;
    }

    size_t tool_count() const {
        return tool_registry_.size();
    }

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
    Agent(const Config& config, std::unique_ptr<core::Model> model)
        : config_(config)
        , model_(std::move(model))
        , request_queue_(config.request_queue_capacity)
        , running_(true)
    {
        inference_thread_ = std::thread([this]() { inference_loop(); });
    }

    void inference_loop() {
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

        // Drain remaining requests
        while (auto remaining = request_queue_.pop()) {
            if (remaining->promise) {
                remaining->promise->set_value(std::unexpected(Error{
                    ErrorCode::AgentNotRunning,
                    "Agent stopped before request could be processed"
                }));
            }
            cleanup_cancel_token(remaining->id);
        }
    }

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

            auto generated = model_->generate_from_history(std::move(callback));

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

    void cleanup_cancel_token(RequestId id) {
        std::lock_guard<std::mutex> lock(cancel_tokens_mutex_);
        cancel_tokens_.erase(id);
    }

    void update_tool_grammar() {
        std::lock_guard<std::mutex> lock(model_mutex_);
        auto schemas = tool_registry_.get_all_schemas();
        auto grammar = tools::GrammarBuilder::build(schemas);
        if (grammar.empty()) {
            model_->clear_tool_grammar();
        } else {
            model_->set_tool_grammar(grammar);
            ZOO_LOG("info", "tool grammar updated (%zu tools)", tool_registry_.size());
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
