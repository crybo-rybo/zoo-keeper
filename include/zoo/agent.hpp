#pragma once

#include "core/types.hpp"
#include "core/model.hpp"
#include "tools/registry.hpp"
#include "tools/parser.hpp"
#include "tools/validation.hpp"
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
    static Expected<std::unique_ptr<Agent>> create(
        const Config& config,
        std::unique_ptr<core::IBackend> backend = nullptr
    ) {
        auto model_result = core::Model::load(config, std::move(backend));
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
        return tool_registry_.register_tool(name, description, param_names, std::move(func));
    }

    size_t tool_count() const {
        return tool_registry_.size();
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
                if (promise) {
                    promise->set_value(std::unexpected(Error{
                        ErrorCode::RequestCancelled, "Request cancelled"
                    }));
                }
                cleanup_cancel_token(request_opt->id);
                continue;
            }

            // Process request (all model access under lock)
            Expected<Response> result = process_request(*request_opt);

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

        tools::ErrorRecovery error_recovery;
        int iteration = 0;
        constexpr int max_tool_iterations = 5;

        while (iteration < max_tool_iterations) {
            ++iteration;

            // Check cancellation
            if (request.cancelled && request.cancelled->load(std::memory_order_acquire)) {
                return std::unexpected(Error{
                    ErrorCode::RequestCancelled,
                    "Request cancelled during tool loop"
                });
            }

            // Build streaming callback
            int completion_tokens = 0;
            auto wrapped_callback = [&](std::string_view token) {
                if (!first_token_received) {
                    first_token_time = std::chrono::steady_clock::now();
                    first_token_received = true;
                }
                ++completion_tokens;
                if (request.streaming_callback) {
                    (*request.streaming_callback)(token);
                }
            };

            auto generated = model_->generate_from_history(
                std::optional<std::function<void(std::string_view)>>(wrapped_callback)
            );

            if (!generated) {
                return std::unexpected(generated.error());
            }

            total_completion_tokens += completion_tokens;

            std::string generated_text = std::move(*generated);

            // Check for tool calls
            if (tool_registry_.size() > 0) {
                auto parse_result = tools::ToolCallParser::parse(generated_text);

                if (parse_result.tool_call.has_value()) {
                    const auto& tc = *parse_result.tool_call;

                    // Commit assistant message with tool call
                    model_->add_message(Message::assistant(generated_text));
                    model_->backend().finalize_response(model_->get_history());

                    // Validate arguments
                    auto validation_error = error_recovery.validate_args(tc, tool_registry_);
                    if (!validation_error.empty()) {
                        if (!error_recovery.can_retry(tc.name)) {
                            return std::unexpected(Error{
                                ErrorCode::ToolRetriesExhausted,
                                "Tool retries exhausted for '" + tc.name + "': " + validation_error
                            });
                        }
                        error_recovery.record_retry(tc.name);

                        std::string error_content = "Error: " + validation_error;
                        model_->add_message(Message::tool(
                            error_content + "\nPlease correct the arguments.", tc.id));
                        tool_call_history.push_back(
                            Message::tool(std::move(error_content), tc.id));
                        continue;
                    }

                    // Execute tool
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
            }

            // No tool call — final response
            auto end_time = std::chrono::steady_clock::now();

            // Commit assistant response
            model_->add_message(Message::assistant(generated_text));
            model_->backend().finalize_response(model_->get_history());

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
