/**
 * @file runtime.cpp
 * @brief Internal worker-owned runtime behind the public Agent facade.
 */

#include "zoo/internal/agent/runtime.hpp"

#include "zoo/internal/log.hpp"
#include "zoo/internal/tools/grammar.hpp"
#include "zoo/internal/tools/interceptor.hpp"
#include "zoo/tools/parser.hpp"
#include "zoo/tools/registry.hpp"
#include "zoo/tools/validation.hpp"
#include <cassert>
#include <chrono>
#include <functional>
#include <thread>
#include <unordered_map>
#include <variant>

namespace zoo::internal::agent {

namespace {

class ScopeExit {
  public:
    explicit ScopeExit(std::function<void()> callback) : callback_(std::move(callback)) {}

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&&) noexcept = default;
    ScopeExit& operator=(ScopeExit&&) noexcept = default;

    ~ScopeExit() {
        if (callback_) {
            callback_();
        }
    }

  private:
    std::function<void()> callback_;
};

Expected<void> load_history(AgentBackend& backend, const std::vector<Message>& messages) {
    backend.clear_history();
    for (const auto& message : messages) {
        if (auto add_result = backend.add_message(message); !add_result) {
            return std::unexpected(add_result.error());
        }
    }
    return {};
}

void restore_history(AgentBackend& backend, std::vector<Message> messages) {
    // Use replace_messages rather than clear_history() + add_message() to avoid
    // a redundant KV-cache flush.  The next generation will re-render from
    // position zero and naturally overwrite any stale entries left by the
    // scoped inference; entries beyond the restored prompt length are harmless
    // because causal attention never reaches them.
    backend.replace_messages(std::move(messages));
}

} // namespace

AgentRuntime::AgentRuntime(const Config& cfg, std::unique_ptr<AgentBackend> backend)
    : config_(cfg), backend_(std::move(backend)), request_mailbox_(cfg.request_queue_capacity) {
    inference_thread_ = std::thread([this]() { inference_loop(); });
}

AgentRuntime::~AgentRuntime() {
    stop();
}

RequestHandle AgentRuntime::chat(Message message,
                                 std::optional<std::function<void(std::string_view)>> callback) {
    auto prepared = request_tracker_.prepare(std::move(message), std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

RequestHandle
AgentRuntime::complete(std::vector<Message> messages,
                       std::optional<std::function<void(std::string_view)>> callback) {
    auto prepared =
        request_tracker_.prepare(std::move(messages), HistoryMode::Replace, std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (prepared.request.messages.empty()) {
        request_tracker_.fail(handle.id, Error{ErrorCode::InvalidMessageSequence,
                                               "Request must include at least one message"});
        return handle;
    }

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

RequestHandle AgentRuntime::extract(const nlohmann::json& output_schema, Message message,
                                    std::optional<std::function<void(std::string_view)>> callback) {
    // Validate schema upfront on the calling thread (fail fast)
    auto params = tools::detail::normalize_schema(output_schema);
    if (!params) {
        auto prepared = request_tracker_.prepare(std::move(message), std::move(callback));
        RequestHandle handle{prepared.request.id, std::move(prepared.future)};
        request_tracker_.fail(handle.id,
                              Error{ErrorCode::InvalidOutputSchema, params.error().message});
        return handle;
    }

    auto prepared = request_tracker_.prepare(std::move(message), nlohmann::json(output_schema),
                                             std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

RequestHandle AgentRuntime::extract(const nlohmann::json& output_schema,
                                    std::vector<Message> messages,
                                    std::optional<std::function<void(std::string_view)>> callback) {
    // Validate schema upfront on the calling thread (fail fast)
    auto params = tools::detail::normalize_schema(output_schema);
    if (!params) {
        auto prepared = request_tracker_.prepare(std::move(messages), HistoryMode::Replace,
                                                 std::move(callback));
        RequestHandle handle{prepared.request.id, std::move(prepared.future)};
        request_tracker_.fail(handle.id,
                              Error{ErrorCode::InvalidOutputSchema, params.error().message});
        return handle;
    }

    auto prepared = request_tracker_.prepare(std::move(messages), HistoryMode::Replace,
                                             nlohmann::json(output_schema), std::move(callback));
    RequestHandle handle{prepared.request.id, std::move(prepared.future)};

    if (prepared.request.messages.empty()) {
        request_tracker_.fail(handle.id, Error{ErrorCode::InvalidMessageSequence,
                                               "Request must include at least one message"});
        return handle;
    }

    if (!running_.load(std::memory_order_acquire)) {
        request_tracker_.fail(handle.id, Error{ErrorCode::AgentNotRunning, "Agent is not running"});
        return handle;
    }

    if (!request_mailbox_.push_request(std::move(prepared.request))) {
        request_tracker_.fail(handle.id, Error{ErrorCode::QueueFull,
                                               "Request queue is full or agent is shutting down"});
        return handle;
    }

    return handle;
}

void AgentRuntime::cancel(RequestId id) {
    request_tracker_.cancel(id);
}

void AgentRuntime::set_system_prompt(const std::string& prompt) {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(SetSystemPromptCmd{prompt, std::move(done)})) {
        return;
    }
    future.get();
}

void AgentRuntime::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
}

bool AgentRuntime::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

std::vector<Message> AgentRuntime::get_history() const {
    if (!running_.load(std::memory_order_acquire)) {
        return {};
    }

    auto done = std::make_shared<std::promise<std::vector<Message>>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(GetHistoryCmd{std::move(done)})) {
        return {};
    }
    return future.get();
}

void AgentRuntime::clear_history() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(ClearHistoryCmd{std::move(done)})) {
        return;
    }
    future.get();
}

Expected<void> AgentRuntime::register_tool(tools::ToolDefinition definition) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());

    if (auto result = tool_registry_.register_tool(std::move(definition)); !result) {
        return std::unexpected(result.error());
    }

    update_tool_grammar();
    return {};
}

size_t AgentRuntime::tool_count() const noexcept {
    return tool_registry_.size();
}

std::string AgentRuntime::build_tool_system_prompt(const std::string& base_prompt) const {
    auto schemas = tool_registry_.get_all_schemas();
    if (schemas.empty()) {
        return base_prompt;
    }

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

void AgentRuntime::inference_loop() {
    try {
        while (running_.load(std::memory_order_acquire)) {
            auto item_opt = request_mailbox_.pop();
            if (!item_opt) {
                break;
            }

            std::visit(overloaded{
                           [this](Request& request) { handle_request(request); },
                           [this](Command& cmd) { handle_command(cmd); },
                       },
                       *item_opt);
        }

        fail_pending(
            Error{ErrorCode::AgentNotRunning, "Agent stopped before request could be processed"});
    } catch (const std::exception& e) {
        ZOO_LOG("error", "fatal exception escaped inference thread: %s", e.what());
        fail_pending(Error{ErrorCode::InferenceFailed,
                           std::string("Inference thread terminated unexpectedly: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "fatal unknown exception escaped inference thread");
        fail_pending(Error{ErrorCode::InferenceFailed, "Inference thread terminated unexpectedly"});
    }
}

void AgentRuntime::handle_request(Request& request) {
    auto promise = request.promise;

    if (request.cancelled && request.cancelled->load(std::memory_order_acquire)) {
        ZOO_LOG("info", "request %lu cancelled before processing",
                static_cast<unsigned long>(request.id));
        request_tracker_.fail(request.id, Error{ErrorCode::RequestCancelled, "Request cancelled"});
        return;
    }

    Expected<Response> result;
    try {
        result = process_request(request);
    } catch (const std::exception& e) {
        ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
        result = std::unexpected(
            Error{ErrorCode::InferenceFailed, std::string("Unhandled exception: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "unknown exception in inference thread");
        result = std::unexpected(
            Error{ErrorCode::InferenceFailed, "Unknown exception in inference thread"});
    }

    request_tracker_.cleanup(request.id);

    if (promise) {
        promise->set_value(std::move(result));
    }
}

void AgentRuntime::handle_command(Command& cmd) {
    std::visit(overloaded{
                   [this](SetSystemPromptCmd& c) {
                       backend_->set_system_prompt(c.prompt);
                       c.done->set_value();
                   },
                   [this](GetHistoryCmd& c) { c.done->set_value(backend_->get_history()); },
                   [this](ClearHistoryCmd& c) {
                       backend_->clear_history();
                       c.done->set_value();
                   },
                   [this](RefreshToolGrammarCmd& c) {
                       auto grammar = tools::GrammarBuilder::build(c.metadata);
                       bool active = false;
                       if (grammar.empty()) {
                           backend_->clear_tool_grammar();
                       } else if (backend_->set_tool_grammar(grammar)) {
                           active = true;
                           ZOO_LOG("info", "tool grammar updated (%zu tools)", c.metadata.size());
                       } else {
                           backend_->clear_tool_grammar();
                           ZOO_LOG("warn", "grammar sampler init failed, falling back to "
                                           "unconstrained generation");
                       }
                       tool_grammar_active_.store(active, std::memory_order_release);
                       c.done->set_value(active);
                   },
               },
               cmd);
}

Expected<Response> AgentRuntime::process_request(const Request& request) {
    if (request.extraction_schema.has_value()) {
        return process_extraction_request(request);
    }

    auto start_time = std::chrono::steady_clock::now();

    std::optional<std::vector<Message>> original_history;
    std::optional<ScopeExit> restore_history_guard;

    if (request.history_mode == HistoryMode::Replace) {
        original_history = backend_->get_history();
        restore_history_guard.emplace(
            [this, &original_history] { restore_history(*backend_, *original_history); });

        if (auto load_result = load_history(*backend_, request.messages); !load_result) {
            return std::unexpected(load_result.error());
        }
    } else {
        if (request.messages.size() != 1u) {
            return std::unexpected(
                Error{ErrorCode::InvalidMessageSequence,
                      "Stateful chat requests must include exactly one message"});
        }

        auto add_result = backend_->add_message(request.messages.front());
        if (!add_result) {
            return std::unexpected(add_result.error());
        }
    }

    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int total_completion_tokens = 0;
    int total_prompt_tokens = 0;
    std::vector<ToolInvocation> tool_invocations;
    std::unordered_map<std::string, int> retry_counts;
    tools::ToolArgumentsValidator validator;
    int iteration = 0;
    const int max_tool_iterations = config_.max_tool_iterations;
    const bool has_tools = tool_registry_.size() > 0;
    const bool use_grammar_path = has_tools && tool_grammar_active_.load(std::memory_order_acquire);

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
            return
                [&, callback = std::move(inner_callback)](std::string_view token) -> TokenAction {
                    if (!first_token_received) {
                        first_token_time = std::chrono::steady_clock::now();
                        first_token_received = true;
                    }
                    ++completion_tokens;
                    return callback(token);
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
            callback = make_metrics_callback(interceptor->make_callback());
        } else if (request.streaming_callback) {
            callback = make_metrics_callback([&](std::string_view token) -> TokenAction {
                (*request.streaming_callback)(token);
                return TokenAction::Continue;
            });
        } else {
            callback = make_metrics_callback(
                [](std::string_view) -> TokenAction { return TokenAction::Continue; });
        }

        auto generated = backend_->generate_from_history(std::move(callback), [&request]() {
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
            const auto& tool_call = *detected_tool_call;
            backend_->add_message(
                Message::assistant(use_grammar_path ? generated->text : response_text));
            backend_->finalize_response();

            std::string args_json = tool_call.arguments.dump();
            if (auto validation_result = validator.validate(tool_call, tool_registry_);
                !validation_result) {
                const Error validation_error = validation_result.error();
                auto& retry_count = retry_counts[tool_call.name];

                if (retry_count >= config_.max_tool_retries) {
                    ZOO_LOG("error", "tool retries exhausted for '%s': %s", tool_call.name.c_str(),
                            validation_error.message.c_str());
                    return std::unexpected(Error{ErrorCode::ToolRetriesExhausted,
                                                 "Tool retries exhausted for '" + tool_call.name +
                                                     "': " + validation_error.message});
                }

                ++retry_count;
                ZOO_LOG("warn", "tool '%s' validation failed (retry %d/%d): %s",
                        tool_call.name.c_str(), retry_count, config_.max_tool_retries,
                        validation_error.message.c_str());

                std::string error_content = "Error: " + validation_error.message;
                backend_->add_message(
                    Message::tool(error_content + "\nPlease correct the arguments.", tool_call.id));
                tool_invocations.push_back(ToolInvocation{
                    tool_call.id, tool_call.name, std::move(args_json),
                    ToolInvocationStatus::ValidationFailed, std::nullopt, validation_error});
                continue;
            }

            ZOO_LOG("info", "invoking tool '%s' (iteration %d, grammar=%d)", tool_call.name.c_str(),
                    iteration, use_grammar_path);
            auto invoke_result = tool_registry_.invoke(tool_call.name, tool_call.arguments);
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

            backend_->add_message(Message::tool(std::move(tool_result_str), tool_call.id));
            tool_invocations.push_back(
                ToolInvocation{tool_call.id, tool_call.name, std::move(args_json), status,
                               std::move(result_json), std::move(tool_error)});
            continue;
        }

        if (response_text.empty() && !tool_invocations.empty() && iteration < max_tool_iterations) {
            backend_->add_message(
                Message::user("Please respond to the user with the tool result."));
            continue;
        }

        auto end_time = std::chrono::steady_clock::now();

        backend_->add_message(Message::assistant(response_text));
        backend_->finalize_response();

        Response response;
        response.text = std::move(response_text);
        response.tool_invocations = std::move(tool_invocations);
        response.usage.prompt_tokens = total_prompt_tokens;
        response.usage.completion_tokens = total_completion_tokens;
        response.usage.total_tokens = total_prompt_tokens + total_completion_tokens;

        response.metrics.latency_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (first_token_received) {
            response.metrics.time_to_first_token_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time -
                                                                      start_time);
            auto generation_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - first_token_time);
            if (generation_time.count() > 0) {
                response.metrics.tokens_per_second =
                    (total_completion_tokens * 1000.0) / generation_time.count();
            }
        }

        return response;
    }

    ZOO_LOG("error", "tool loop iteration limit reached (%d)", max_tool_iterations);
    return std::unexpected(
        Error{ErrorCode::ToolLoopLimitReached,
              "Tool loop iteration limit reached (" + std::to_string(max_tool_iterations) + ")"});
}

Expected<Response> AgentRuntime::process_extraction_request(const Request& request) {
    auto start_time = std::chrono::steady_clock::now();

    // Normalize schema and build GBNF grammar
    auto params = tools::detail::normalize_schema(*request.extraction_schema);
    if (!params) {
        return std::unexpected(Error{ErrorCode::InvalidOutputSchema, params.error().message});
    }

    std::string grammar_str = tools::GrammarBuilder::build_schema(*params);

    // Save/restore history for stateless (Replace) mode
    std::optional<std::vector<Message>> original_history;
    std::optional<ScopeExit> restore_history_guard;

    if (request.history_mode == HistoryMode::Replace) {
        original_history = backend_->get_history();
        restore_history_guard.emplace(
            [this, &original_history] { restore_history(*backend_, *original_history); });

        if (auto load_result = load_history(*backend_, request.messages); !load_result) {
            return std::unexpected(load_result.error());
        }
    } else {
        if (request.messages.size() != 1u) {
            return std::unexpected(
                Error{ErrorCode::InvalidMessageSequence,
                      "Stateful extraction requests must include exactly one message"});
        }

        auto add_result = backend_->add_message(request.messages.front());
        if (!add_result) {
            return std::unexpected(add_result.error());
        }
    }

    // Set schema grammar, restore previous grammar on exit
    const bool had_tool_grammar = tool_grammar_active_.load(std::memory_order_acquire);

    if (!backend_->set_schema_grammar(grammar_str)) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed, "Failed to initialize schema grammar"});
    }

    ScopeExit grammar_guard([this, had_tool_grammar] {
        backend_->clear_tool_grammar();
        if (had_tool_grammar) {
            auto metadata = tool_registry_.get_all_tool_metadata();
            auto tool_grammar = tools::GrammarBuilder::build(metadata);
            bool restored = false;
            if (!tool_grammar.empty()) {
                restored = backend_->set_tool_grammar(tool_grammar);
            }
            tool_grammar_active_.store(restored, std::memory_order_release);
        }
    });

    // Generate (single pass, no tool loop)
    std::chrono::steady_clock::time_point first_token_time;
    bool first_token_received = false;
    int completion_tokens = 0;

    std::optional<TokenCallback> callback;
    if (request.streaming_callback) {
        callback = [&](std::string_view token) -> TokenAction {
            if (!first_token_received) {
                first_token_time = std::chrono::steady_clock::now();
                first_token_received = true;
            }
            ++completion_tokens;
            (*request.streaming_callback)(token);
            return TokenAction::Continue;
        };
    } else {
        callback = [&](std::string_view) -> TokenAction {
            if (!first_token_received) {
                first_token_time = std::chrono::steady_clock::now();
                first_token_received = true;
            }
            ++completion_tokens;
            return TokenAction::Continue;
        };
    }

    auto generated = backend_->generate_from_history(std::move(callback), [&request]() {
        return request.cancelled && request.cancelled->load(std::memory_order_acquire);
    });
    if (!generated) {
        return std::unexpected(generated.error());
    }

    // Parse and validate extracted JSON
    nlohmann::json extracted;
    try {
        extracted = nlohmann::json::parse(generated->text);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed,
                  std::string("Failed to parse extraction output as JSON: ") + e.what()});
    }

    if (auto validation = tools::validate_json_against_schema(extracted, *params); !validation) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed,
                  "Extracted JSON failed schema validation: " + validation.error().message});
    }

    // Commit the assistant response to history
    backend_->add_message(Message::assistant(generated->text));
    backend_->finalize_response();

    auto end_time = std::chrono::steady_clock::now();

    Response response;
    response.text = std::move(generated->text);
    response.extracted_data = std::move(extracted);
    response.usage.prompt_tokens = generated->prompt_tokens;
    response.usage.completion_tokens = completion_tokens;
    response.usage.total_tokens = generated->prompt_tokens + completion_tokens;

    response.metrics.latency_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (first_token_received) {
        response.metrics.time_to_first_token_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        auto generation_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - first_token_time);
        if (generation_time.count() > 0) {
            response.metrics.tokens_per_second =
                (completion_tokens * 1000.0) / generation_time.count();
        }
    }

    return response;
}

void AgentRuntime::fail_pending(const Error& error) {
    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();

    while (auto remaining = request_mailbox_.pop()) {
        std::visit(overloaded{
                       [&](Request& request) { request_tracker_.fail(request.id, error); },
                       [](Command& cmd) { resolve_command_on_shutdown(cmd); },
                   },
                   *remaining);
    }

    request_tracker_.fail_all(error);
}

void AgentRuntime::resolve_command_on_shutdown(Command& cmd) {
    std::visit(overloaded{
                   [](SetSystemPromptCmd& c) { c.done->set_value(); },
                   [](GetHistoryCmd& c) { c.done->set_value(std::vector<Message>{}); },
                   [](ClearHistoryCmd& c) { c.done->set_value(); },
                   [](RefreshToolGrammarCmd& c) { c.done->set_value(false); },
               },
               cmd);
}

void AgentRuntime::update_tool_grammar() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    auto metadata = tool_registry_.get_all_tool_metadata();
    auto done = std::make_shared<std::promise<bool>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(
            RefreshToolGrammarCmd{std::move(metadata), std::move(done)})) {
        return;
    }
    future.get();
}

} // namespace zoo::internal::agent
