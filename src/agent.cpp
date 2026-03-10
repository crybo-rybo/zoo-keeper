/**
 * @file agent.cpp
 * @brief Implementation of the asynchronous agent orchestration layer.
 */

#include "zoo/agent.hpp"

#include "zoo/core/model.hpp"
#include "zoo/internal/agent/mailbox.hpp"
#include "zoo/internal/agent/request_tracker.hpp"
#include "zoo/internal/log.hpp"
#include "zoo/internal/tools/grammar.hpp"
#include "zoo/internal/tools/interceptor.hpp"
#include "zoo/tools/parser.hpp"
#include "zoo/tools/validation.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <variant>

namespace zoo {
namespace runtime = internal::agent;

struct Agent::Impl {
    explicit Impl(const Config& cfg, std::unique_ptr<core::Model> owned_model)
        : config(cfg), model(std::move(owned_model)),
          request_mailbox(cfg.request_queue_capacity),
          running(true) {
        inference_thread = std::thread([this]() { inference_loop(); });
    }

    ~Impl() {
        stop();
    }

    RequestHandle
    chat(Message message,
         std::optional<std::function<void(std::string_view)>> callback = std::nullopt) {
        auto prepared = request_tracker.prepare(std::move(message), std::move(callback));
        RequestHandle handle{prepared.request.id, std::move(prepared.future)};

        if (!running.load(std::memory_order_acquire)) {
            request_tracker.fail(handle.id,
                                 Error{ErrorCode::AgentNotRunning, "Agent is not running"});
            return handle;
        }

        if (!request_mailbox.push_request(std::move(prepared.request))) {
            request_tracker.fail(
                handle.id,
                Error{ErrorCode::QueueFull, "Request queue is full or agent is shutting down"});
            return handle;
        }

        return handle;
    }

    void cancel(RequestId id) {
        request_tracker.cancel(id);
    }

    void set_system_prompt(const std::string& prompt) {
        if (!running.load(std::memory_order_acquire))
            return;
        auto done = std::make_shared<std::promise<void>>();
        auto future = done->get_future();
        if (!request_mailbox.push_command(
                runtime::SetSystemPromptCmd{prompt, std::move(done)})) {
            return;
        }
        future.get();
    }

    void stop() {
        if (!running.load(std::memory_order_acquire))
            return;
        running.store(false, std::memory_order_release);
        request_mailbox.shutdown();
        if (inference_thread.joinable()) {
            inference_thread.join();
        }
    }

    bool is_running() const noexcept {
        return running.load(std::memory_order_acquire);
    }

    std::vector<Message> get_history() {
        if (!running.load(std::memory_order_acquire))
            return {};
        auto done = std::make_shared<std::promise<std::vector<Message>>>();
        auto future = done->get_future();
        if (!request_mailbox.push_command(runtime::GetHistoryCmd{std::move(done)})) {
            return {};
        }
        return future.get();
    }

    void clear_history() {
        if (!running.load(std::memory_order_acquire))
            return;
        auto done = std::make_shared<std::promise<void>>();
        auto future = done->get_future();
        if (!request_mailbox.push_command(runtime::ClearHistoryCmd{std::move(done)})) {
            return;
        }
        future.get();
    }

    Expected<void> register_tool(tools::ToolDefinition definition) {
        if (auto result = tool_registry.register_tool(std::move(definition)); !result) {
            return std::unexpected(result.error());
        }
        update_tool_grammar();
        return {};
    }

    size_t tool_count() const noexcept {
        return tool_registry.size();
    }

    std::string build_tool_system_prompt(const std::string& base_prompt) const {
        auto schemas = tool_registry.get_all_schemas();
        if (schemas.empty())
            return base_prompt;

        if (tool_grammar_active_.load(std::memory_order_acquire)) {
            return base_prompt +
                   "\n\nYou have access to tools. When you need to use a tool, wrap the "
                   "call in sentinel tags like this:\n"
                   "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param1\": "
                   "\"value1\"}}</tool_call>\n"
                   "\nYou may think step-by-step before calling a tool. "
                   "Do NOT output any text after the </tool_call> closing tag.\n"
                   "After receiving a tool result, incorporate it into a natural response.\n"
                   "If no tool is needed, respond normally without sentinel tags.\n"
                   "\nAvailable tools:\n" +
                   schemas.dump(2);
        }

        return base_prompt +
               "\n\nWhen you need to use a tool, respond with a JSON object containing "
               "\"name\" and \"arguments\" fields. For example:\n"
               "{\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n"
               "\nOutput ONLY the JSON tool call when invoking a tool — no text after it.\n"
               "After receiving a tool result, incorporate it into a natural response.\n"
               "If no tool is needed, respond normally without JSON.\n"
               "\nAvailable tools:\n" +
               schemas.dump(2);
    }

    // -----------------------------------------------------------------------
    // Inference thread
    // -----------------------------------------------------------------------

    void inference_loop() {
        try {
            while (running.load(std::memory_order_acquire)) {
                auto item_opt = request_mailbox.pop();
                if (!item_opt)
                    break;

                std::visit(
                    [this](auto&& item) {
                        using T = std::decay_t<decltype(item)>;
                        if constexpr (std::is_same_v<T, runtime::Request>) {
                            handle_request(item);
                        } else if constexpr (std::is_same_v<T, runtime::Command>) {
                            handle_command(item);
                        }
                    },
                    *item_opt);
            }

            fail_pending(Error{ErrorCode::AgentNotRunning,
                               "Agent stopped before request could be processed"});
        } catch (const std::exception& e) {
            ZOO_LOG("error", "fatal exception escaped inference thread: %s", e.what());
            fail_pending(
                Error{ErrorCode::InferenceFailed,
                      std::string("Inference thread terminated unexpectedly: ") + e.what()});
        } catch (...) {
            ZOO_LOG("error", "fatal unknown exception escaped inference thread");
            fail_pending(
                Error{ErrorCode::InferenceFailed, "Inference thread terminated unexpectedly"});
        }
    }

    void handle_request(runtime::Request& request) {
        auto promise = request.promise;

        if (request.cancelled && request.cancelled->load(std::memory_order_acquire)) {
            ZOO_LOG("info", "request %lu cancelled before processing",
                    static_cast<unsigned long>(request.id));
            request_tracker.fail(request.id,
                                 Error{ErrorCode::RequestCancelled, "Request cancelled"});
            return;
        }

        Expected<Response> result;
        try {
            result = process_request(request);
        } catch (const std::exception& e) {
            ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
            result = std::unexpected(Error{ErrorCode::InferenceFailed,
                                           std::string("Unhandled exception: ") + e.what()});
        } catch (...) {
            ZOO_LOG("error", "unknown exception in inference thread");
            result = std::unexpected(
                Error{ErrorCode::InferenceFailed, "Unknown exception in inference thread"});
        }

        request_tracker.cleanup(request.id);

        if (promise) {
            promise->set_value(std::move(result));
        }
    }

    void handle_command(runtime::Command& cmd) {
        std::visit(
            [this](auto&& c) {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, runtime::SetSystemPromptCmd>) {
                    model->set_system_prompt(c.prompt);
                    c.done->set_value();
                } else if constexpr (std::is_same_v<T, runtime::GetHistoryCmd>) {
                    c.done->set_value(model->get_history());
                } else if constexpr (std::is_same_v<T, runtime::ClearHistoryCmd>) {
                    model->clear_history();
                    c.done->set_value();
                } else if constexpr (std::is_same_v<T, runtime::RefreshToolGrammarCmd>) {
                    auto grammar = tools::GrammarBuilder::build(c.metadata);
                    bool active = false;
                    if (grammar.empty()) {
                        model->clear_tool_grammar();
                    } else if (model->set_tool_grammar(grammar)) {
                        active = true;
                        ZOO_LOG("info", "tool grammar updated (%zu tools)",
                                c.metadata.size());
                    } else {
                        model->clear_tool_grammar();
                        ZOO_LOG("warn",
                                "grammar sampler init failed, falling back to "
                                "unconstrained generation");
                    }
                    tool_grammar_active_.store(active, std::memory_order_release);
                    c.done->set_value(active);
                }
            },
            cmd);
    }

    Expected<Response> process_request(const runtime::Request& request) {
        auto start_time = std::chrono::steady_clock::now();

        auto add_result = model->add_message(request.message);
        if (!add_result) {
            return std::unexpected(add_result.error());
        }

        std::chrono::steady_clock::time_point first_token_time;
        bool first_token_received = false;
        int total_completion_tokens = 0;
        int total_prompt_tokens = 0;
        std::vector<ToolInvocation> tool_invocations;
        std::unordered_map<std::string, int> retry_counts;
        tools::ToolArgumentsValidator validator;
        int iteration = 0;
        const int max_tool_iterations = config.max_tool_iterations;
        const bool has_tools = tool_registry.size() > 0;
        const bool use_grammar_path =
            has_tools && tool_grammar_active_.load(std::memory_order_acquire);

        ZOO_LOG("debug", "processing request %lu (tools=%d, grammar=%d)",
                static_cast<unsigned long>(request.id), has_tools, use_grammar_path);

        while (iteration < max_tool_iterations) {
            ++iteration;

            if (request.cancelled && request.cancelled->load(std::memory_order_acquire)) {
                return std::unexpected(
                    Error{ErrorCode::RequestCancelled, "Request cancelled during tool loop"});
            }

            int completion_tokens = 0;

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

            std::optional<TokenCallback> callback;
            std::optional<tools::ToolCallInterceptor> interceptor;

            if (use_grammar_path) {
                if (request.streaming_callback) {
                    callback = make_metrics_callback([&](std::string_view token) -> TokenAction {
                        (*request.streaming_callback)(token);
                        return TokenAction::Continue;
                    });
                } else {
                    callback = make_metrics_callback(
                        [](std::string_view) -> TokenAction { return TokenAction::Continue; });
                }
            } else if (has_tools) {
                interceptor.emplace(request.streaming_callback);
                auto interceptor_cb = interceptor->make_callback();
                callback = make_metrics_callback(std::move(interceptor_cb));
            } else if (request.streaming_callback) {
                callback = make_metrics_callback([&](std::string_view token) -> TokenAction {
                    (*request.streaming_callback)(token);
                    return TokenAction::Continue;
                });
            } else {
                callback = make_metrics_callback(
                    [](std::string_view) -> TokenAction { return TokenAction::Continue; });
            }

            auto generated = model->generate_from_history(std::move(callback), [&request]() {
                return request.cancelled && request.cancelled->load(std::memory_order_acquire);
            });

            if (!generated) {
                return std::unexpected(generated.error());
            }

            total_completion_tokens += completion_tokens;
            total_prompt_tokens += generated->prompt_tokens;

            std::optional<tools::ToolCall> detected_tool_call;
            std::string response_text;

            if (use_grammar_path && generated->tool_call_detected) {
                auto sentinel_result = tools::ToolCallParser::parse_sentinel(generated->text);
                detected_tool_call = std::move(sentinel_result.tool_call);
                response_text = std::move(sentinel_result.text_before);
            } else if (interceptor) {
                auto intercept_result = interceptor->finalize();
                detected_tool_call = std::move(intercept_result.tool_call);
                response_text = std::move(intercept_result.full_text);
            } else {
                response_text = std::move(generated->text);
            }

            if (detected_tool_call.has_value()) {
                const auto& tc = *detected_tool_call;

                model->add_message(
                    Message::assistant(use_grammar_path ? generated->text : response_text));
                model->finalize_response();

                std::string args_json = tc.arguments.dump();

                if (auto validation_result = validator.validate(tc, tool_registry);
                    !validation_result) {
                    const Error validation_error = validation_result.error();

                    auto& retry_count = retry_counts[tc.name];
                    if (retry_count >= config.max_tool_retries) {
                        ZOO_LOG("error", "tool retries exhausted for '%s': %s", tc.name.c_str(),
                                validation_error.message.c_str());
                        return std::unexpected(Error{ErrorCode::ToolRetriesExhausted,
                                                     "Tool retries exhausted for '" + tc.name +
                                                         "': " + validation_error.message});
                    }

                    ++retry_count;
                    ZOO_LOG("warn", "tool '%s' validation failed (retry %d/%d): %s",
                            tc.name.c_str(), retry_count, config.max_tool_retries,
                            validation_error.message.c_str());

                    std::string error_content = "Error: " + validation_error.message;
                    model->add_message(
                        Message::tool(error_content + "\nPlease correct the arguments.", tc.id));
                    tool_invocations.push_back(ToolInvocation{
                        tc.id, tc.name, args_json, ToolInvocationStatus::ValidationFailed,
                        std::nullopt, validation_error});
                    continue;
                }

                ZOO_LOG("info", "invoking tool '%s' (iteration %d, grammar=%d)", tc.name.c_str(),
                        iteration, use_grammar_path);
                auto invoke_result = tool_registry.invoke(tc.name, tc.arguments);
                std::string tool_result_str;
                std::optional<std::string> result_json;
                std::optional<Error> tool_error;
                ToolInvocationStatus status = ToolInvocationStatus::Succeeded;
                if (invoke_result) {
                    tool_result_str = invoke_result->dump();
                    result_json = tool_result_str;
                } else {
                    tool_result_str = "Error: " + invoke_result.error().message;
                    tool_error = invoke_result.error();
                    status = ToolInvocationStatus::ExecutionFailed;
                }

                Message tool_msg = Message::tool(std::move(tool_result_str), tc.id);
                model->add_message(tool_msg);
                tool_invocations.push_back(ToolInvocation{tc.id, tc.name, std::move(args_json),
                                                          status, std::move(result_json),
                                                          std::move(tool_error)});
                continue;
            }

            if (response_text.empty() && !tool_invocations.empty() &&
                iteration < max_tool_iterations) {
                model->add_message(
                    Message::user("Please respond to the user with the tool result."));
                continue;
            }

            auto end_time = std::chrono::steady_clock::now();

            model->add_message(Message::assistant(response_text));
            model->finalize_response();

            Response response;
            response.text = std::move(response_text);
            response.tool_invocations = std::move(tool_invocations);
            response.usage.prompt_tokens = total_prompt_tokens;
            response.usage.completion_tokens = total_completion_tokens;
            response.usage.total_tokens = total_prompt_tokens + total_completion_tokens;

            auto total_latency =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            response.metrics.latency_ms = total_latency;

            if (first_token_received) {
                response.metrics.time_to_first_token_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time -
                                                                          start_time);
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
        return std::unexpected(
            Error{ErrorCode::ToolLoopLimitReached, "Tool loop iteration limit reached (" +
                                                       std::to_string(max_tool_iterations) + ")"});
    }

    // -----------------------------------------------------------------------
    // Shutdown helpers
    // -----------------------------------------------------------------------

    void fail_pending(const Error& error) {
        running.store(false, std::memory_order_release);
        request_mailbox.shutdown();

        while (auto remaining = request_mailbox.pop()) {
            std::visit(
                [&](auto&& item) {
                    using T = std::decay_t<decltype(item)>;
                    if constexpr (std::is_same_v<T, runtime::Request>) {
                        request_tracker.fail(item.id, error);
                    } else if constexpr (std::is_same_v<T, runtime::Command>) {
                        resolve_command_on_shutdown(item);
                    }
                },
                *remaining);
        }
    }

    static void resolve_command_on_shutdown(runtime::Command& cmd) {
        std::visit(
            [](auto&& c) {
                using T = std::decay_t<decltype(c)>;
                if constexpr (std::is_same_v<T, runtime::SetSystemPromptCmd>) {
                    c.done->set_value();
                } else if constexpr (std::is_same_v<T, runtime::GetHistoryCmd>) {
                    c.done->set_value(std::vector<Message>{});
                } else if constexpr (std::is_same_v<T, runtime::ClearHistoryCmd>) {
                    c.done->set_value();
                } else if constexpr (std::is_same_v<T, runtime::RefreshToolGrammarCmd>) {
                    c.done->set_value(false);
                }
            },
            cmd);
    }

    void update_tool_grammar() {
        if (!running.load(std::memory_order_acquire))
            return;
        auto metadata = tool_registry.get_all_tool_metadata();
        auto done = std::make_shared<std::promise<bool>>();
        auto future = done->get_future();
        if (!request_mailbox.push_command(
                runtime::RefreshToolGrammarCmd{std::move(metadata), std::move(done)})) {
            return;
        }
        future.get();
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    Config config;
    std::unique_ptr<core::Model> model;
    tools::ToolRegistry tool_registry;
    runtime::RequestTracker request_tracker;
    mutable runtime::RuntimeMailbox request_mailbox;

    std::thread inference_thread;
    std::atomic<bool> running;
    std::atomic<bool> tool_grammar_active_{false};
};

Expected<std::unique_ptr<Agent>> Agent::create(const Config& config) {
    auto model_result = core::Model::load(config);
    if (!model_result) {
        return std::unexpected(model_result.error());
    }

    auto agent_impl = std::make_unique<Impl>(config, std::move(*model_result));
    return std::unique_ptr<Agent>(new Agent(config, std::move(agent_impl)));
}

Agent::Agent(Config config, std::unique_ptr<Impl> impl)
    : config_(std::move(config)), impl_(std::move(impl)) {}

Agent::~Agent() = default;

RequestHandle Agent::chat(Message message,
                          std::optional<std::function<void(std::string_view)>> callback) {
    return impl_->chat(std::move(message), std::move(callback));
}

void Agent::cancel(RequestId id) {
    impl_->cancel(id);
}

void Agent::set_system_prompt(const std::string& prompt) {
    impl_->set_system_prompt(prompt);
}

void Agent::stop() {
    impl_->stop();
}

bool Agent::is_running() const noexcept {
    return impl_->is_running();
}

std::vector<Message> Agent::get_history() const {
    return impl_->get_history();
}

void Agent::clear_history() {
    impl_->clear_history();
}

Expected<void> Agent::register_tool(tools::ToolDefinition definition) {
    return impl_->register_tool(std::move(definition));
}

Expected<void> Agent::register_tool(const std::string& name, const std::string& description,
                                    nlohmann::json schema, tools::ToolHandler handler) {
    auto definition =
        tools::detail::make_tool_definition(name, description, schema, std::move(handler));
    if (!definition) {
        return std::unexpected(definition.error());
    }
    return register_tool(std::move(*definition));
}

size_t Agent::tool_count() const noexcept {
    return impl_->tool_count();
}

std::string Agent::build_tool_system_prompt(const std::string& base_prompt) const {
    return impl_->build_tool_system_prompt(base_prompt);
}

} // namespace zoo
