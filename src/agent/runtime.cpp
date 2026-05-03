/**
 * @file runtime.cpp
 * @brief Internal worker-owned runtime: lifecycle, request submission, and dispatch.
 *
 * The agentic tool loop lives in runtime_tool_loop.cpp.
 * Structured output extraction lives in runtime_extraction.cpp.
 */

#include "agent/runtime.hpp"

#include "agent/request_state.hpp"
#include "log.hpp"
#include "zoo/core/model.hpp"
#include "zoo/tools/registry.hpp"
#include <cassert>
#include <future>
#include <string_view>
#include <thread>

namespace zoo::internal::agent {

namespace {

std::vector<Message> materialize_conversation(ConversationView messages) {
    std::vector<Message> owned;
    owned.reserve(messages.size());
    for (size_t index = 0; index < messages.size(); ++index) {
        owned.push_back(Message::from_view(messages[index]));
    }
    return owned;
}

Error command_timeout_error(std::string_view command_name) {
    return Error{ErrorCode::RequestTimeout,
                 "Timed out waiting for command to complete: " + std::string(command_name)};
}

} // namespace

AgentRuntime::AgentRuntime(ModelConfig model_config, AgentConfig agent_config,
                           GenerationOptions default_generation,
                           std::unique_ptr<AgentBackend> backend)
    : model_config_(std::move(model_config)), agent_config_(agent_config),
      default_generation_options_(std::move(default_generation)), backend_(std::move(backend)),
      request_slots_(std::make_shared<RequestSlots>(agent_config_.request_queue_capacity)),
      request_mailbox_() {
    inference_thread_ = std::thread([this]() { inference_loop(); });
}

AgentRuntime::~AgentRuntime() {
    stop();
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

RequestHandle<TextResponse> AgentRuntime::chat(std::string_view user_message,
                                               const GenerationOptions& options,
                                               AsyncTextCallback callback) {
    return chat(MessageView{Role::User, user_message}, options, std::move(callback));
}

RequestHandle<TextResponse> AgentRuntime::chat(MessageView message,
                                               const GenerationOptions& options,
                                               AsyncTextCallback callback) {
    RequestPayload payload;
    payload.messages.push_back(Message::from_view(message));
    payload.history_mode = HistoryMode::Append;
    payload.options = resolve_generation_options(options);
    payload.streaming_callback = std::move(callback);
    payload.result_kind = ResultKind::Text;
    return enqueue_request<TextResponse>(std::move(payload));
}

RequestHandle<TextResponse> AgentRuntime::complete(ConversationView messages,
                                                   const GenerationOptions& options,
                                                   AsyncTextCallback callback) {
    RequestPayload payload;
    payload.messages = materialize_conversation(messages);
    payload.history_mode = HistoryMode::Replace;
    payload.options = resolve_generation_options(options);
    payload.streaming_callback = std::move(callback);
    payload.result_kind = ResultKind::Text;

    if (payload.messages.empty()) {
        return make_immediate_error_handle<TextResponse>(
            Error{ErrorCode::InvalidMessageSequence, "Request must include at least one message"});
    }

    return enqueue_request<TextResponse>(std::move(payload));
}

RequestHandle<ExtractionResponse> AgentRuntime::extract(const nlohmann::json& output_schema,
                                                        std::string_view user_message,
                                                        const GenerationOptions& options,
                                                        AsyncTextCallback callback) {
    return extract(output_schema, MessageView{Role::User, user_message}, options,
                   std::move(callback));
}

RequestHandle<ExtractionResponse> AgentRuntime::extract(const nlohmann::json& output_schema,
                                                        MessageView message,
                                                        const GenerationOptions& options,
                                                        AsyncTextCallback callback) {
    auto params = tools::detail::normalize_schema(output_schema);
    if (!params) {
        return make_immediate_error_handle<ExtractionResponse>(
            Error{ErrorCode::InvalidOutputSchema, params.error().message});
    }

    RequestPayload payload;
    payload.messages.push_back(Message::from_view(message));
    payload.history_mode = HistoryMode::Append;
    payload.options = resolve_generation_options(options);
    payload.streaming_callback = std::move(callback);
    payload.extraction_schema = nlohmann::json(output_schema);
    payload.result_kind = ResultKind::Extraction;
    return enqueue_request<ExtractionResponse>(std::move(payload));
}

RequestHandle<ExtractionResponse> AgentRuntime::extract(const nlohmann::json& output_schema,
                                                        ConversationView messages,
                                                        const GenerationOptions& options,
                                                        AsyncTextCallback callback) {
    auto params = tools::detail::normalize_schema(output_schema);
    if (!params) {
        return make_immediate_error_handle<ExtractionResponse>(
            Error{ErrorCode::InvalidOutputSchema, params.error().message});
    }

    RequestPayload payload;
    payload.messages = materialize_conversation(messages);
    payload.history_mode = HistoryMode::Replace;
    payload.options = resolve_generation_options(options);
    payload.streaming_callback = std::move(callback);
    payload.extraction_schema = nlohmann::json(output_schema);
    payload.result_kind = ResultKind::Extraction;

    if (payload.messages.empty()) {
        return make_immediate_error_handle<ExtractionResponse>(
            Error{ErrorCode::InvalidMessageSequence, "Request must include at least one message"});
    }

    return enqueue_request<ExtractionResponse>(std::move(payload));
}

void AgentRuntime::cancel(RequestId id) {
    request_slots_->cancel(id);
}

template <typename Result, typename Maker>
Expected<Result> AgentRuntime::send_sync_command(Maker&& make_cmd,
                                                 std::optional<std::chrono::nanoseconds> timeout,
                                                 std::string_view name) {
    if (!running_.load(std::memory_order_acquire)) {
        return std::unexpected(Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }

    auto done = std::make_shared<std::promise<Expected<Result>>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(std::forward<Maker>(make_cmd)(std::move(done)))) {
        return std::unexpected(Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }
    if (timeout && future.wait_for(*timeout) != std::future_status::ready) {
        return std::unexpected(command_timeout_error(name));
    }
    return future.get();
}

namespace {

template <typename CmdT> auto make_string_cmd(std::string s) {
    return [s = std::move(s)](auto done) mutable -> Command {
        return CmdT{std::move(s), std::move(done)};
    };
}

} // namespace

Expected<void>
AgentRuntime::set_system_prompt_impl(std::string prompt,
                                     std::optional<std::chrono::nanoseconds> timeout) {
    return send_sync_command<void>(make_string_cmd<SetSystemPromptCmd>(std::move(prompt)), timeout,
                                   "set_system_prompt");
}

void AgentRuntime::set_system_prompt(std::string_view prompt) {
    (void)set_system_prompt_impl(std::string(prompt), std::nullopt);
}

Expected<void> AgentRuntime::set_system_prompt(std::string_view prompt,
                                               std::chrono::nanoseconds timeout) {
    return set_system_prompt_impl(std::string(prompt), timeout);
}

Expected<void>
AgentRuntime::add_system_message_impl(std::string message,
                                      std::optional<std::chrono::nanoseconds> timeout) {
    return send_sync_command<void>(make_string_cmd<AddSystemMessageCmd>(std::move(message)),
                                   timeout, "add_system_message");
}

Expected<void> AgentRuntime::add_system_message(std::string_view message) {
    return add_system_message_impl(std::string(message), std::nullopt);
}

Expected<void> AgentRuntime::add_system_message(std::string_view message,
                                                std::chrono::nanoseconds timeout) {
    return add_system_message_impl(std::string(message), timeout);
}

Expected<HistorySnapshot>
AgentRuntime::get_history_impl(std::optional<std::chrono::nanoseconds> timeout) const {
    return const_cast<AgentRuntime*>(this)->send_sync_command<HistorySnapshot>(
        [](auto done) -> Command { return GetHistoryCmd{std::move(done)}; }, timeout,
        "get_history");
}

HistorySnapshot AgentRuntime::get_history() const {
    return get_history_impl(std::nullopt).value_or(HistorySnapshot{});
}

Expected<HistorySnapshot> AgentRuntime::get_history(std::chrono::nanoseconds timeout) const {
    return get_history_impl(timeout);
}

Expected<void> AgentRuntime::clear_history_impl(std::optional<std::chrono::nanoseconds> timeout) {
    return send_sync_command<void>(
        [](auto done) -> Command { return ClearHistoryCmd{std::move(done)}; }, timeout,
        "clear_history");
}

void AgentRuntime::clear_history() {
    (void)clear_history_impl(std::nullopt);
}

Expected<void> AgentRuntime::clear_history(std::chrono::nanoseconds timeout) {
    return clear_history_impl(timeout);
}

Expected<void> AgentRuntime::register_tool_impl(tools::ToolDefinition definition,
                                                std::optional<std::chrono::nanoseconds> timeout) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());
    return send_sync_command<void>(
        [d = std::move(definition)](auto done) mutable -> Command {
            return RegisterToolCmd{std::move(d), std::move(done)};
        },
        timeout, "register_tool");
}

Expected<void> AgentRuntime::register_tool(tools::ToolDefinition definition) {
    return register_tool_impl(std::move(definition), std::nullopt);
}

Expected<void> AgentRuntime::register_tool(tools::ToolDefinition definition,
                                           std::chrono::nanoseconds timeout) {
    return register_tool_impl(std::move(definition), timeout);
}

Expected<void> AgentRuntime::register_tools_impl(std::vector<tools::ToolDefinition> definitions,
                                                 std::optional<std::chrono::nanoseconds> timeout) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());
    if (definitions.empty()) {
        return {};
    }
    return send_sync_command<void>(
        [d = std::move(definitions)](auto done) mutable -> Command {
            return RegisterToolsCmd{std::move(d), std::move(done)};
        },
        timeout, "register_tools");
}

Expected<void> AgentRuntime::register_tools(std::vector<tools::ToolDefinition> definitions) {
    return register_tools_impl(std::move(definitions), std::nullopt);
}

Expected<void> AgentRuntime::register_tools(std::vector<tools::ToolDefinition> definitions,
                                            std::chrono::nanoseconds timeout) {
    return register_tools_impl(std::move(definitions), timeout);
}

size_t AgentRuntime::tool_count() const noexcept {
    return tool_registry_.size();
}

void AgentRuntime::inference_loop() {
    try {
        while (running_.load(std::memory_order_acquire)) {
            auto item_opt = request_mailbox_.pop();
            if (!item_opt) {
                break;
            }

            std::visit(overloaded{
                           [this](QueuedRequest request) { handle_request(request); },
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

void AgentRuntime::handle_request(QueuedRequest request) {
    const auto active_request = request_slots_->active_request(request);
    if (!active_request.has_value()) {
        return;
    }

    if (active_request->cancelled && active_request->cancelled->load(std::memory_order_acquire)) {
        request_slots_->resolve_error(
            request.slot, request.generation,
            Error{ErrorCode::RequestCancelled, "Request cancelled before processing"});
        return;
    }

    try {
        if (active_request->result_kind == ResultKind::Extraction) {
            request_slots_->resolve_extraction(request.slot, request.generation,
                                               process_extraction_request(*active_request));
        } else {
            request_slots_->resolve_text(request.slot, request.generation,
                                         process_request(*active_request));
        }
    } catch (const std::exception& e) {
        ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
        request_slots_->resolve_error(
            request.slot, request.generation,
            Error{ErrorCode::InferenceFailed, std::string("Unhandled exception: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "unknown exception in inference thread");
        request_slots_->resolve_error(
            request.slot, request.generation,
            Error{ErrorCode::InferenceFailed, "Unknown exception in inference thread"});
    }
}

void AgentRuntime::handle_command(Command& cmd) {
    std::visit(
        overloaded{
            [this](SetSystemPromptCmd& c) {
                backend_->set_system_prompt(c.prompt);
                c.done->set_value(Expected<void>{});
            },
            [this](GetHistoryCmd& c) {
                c.done->set_value(Expected<HistorySnapshot>{backend_->get_history()});
            },
            [this](ClearHistoryCmd& c) {
                backend_->clear_history();
                c.done->set_value(Expected<void>{});
            },
            [this](AddSystemMessageCmd& c) {
                c.done->set_value(backend_->add_message(MessageView{Role::System, c.message}));
            },
            [this](RegisterToolCmd& c) {
                if (auto result = tool_registry_.register_tool(std::move(c.definition)); !result) {
                    c.done->set_value(std::unexpected(result.error()));
                    return;
                }

                refresh_tool_calling_state();
                c.done->set_value({});
            },
            [this](RegisterToolsCmd& c) {
                if (auto result = tool_registry_.register_tools(std::move(c.definitions));
                    !result) {
                    c.done->set_value(std::unexpected(result.error()));
                    return;
                }

                refresh_tool_calling_state();
                c.done->set_value({});
            },
        },
        cmd);
}

void AgentRuntime::fail_pending(const Error& error) {
    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();

    while (auto remaining = request_mailbox_.pop()) {
        std::visit(overloaded{
                       [&](QueuedRequest request) {
                           request_slots_->resolve_error(request.slot, request.generation, error);
                       },
                       [](Command& cmd) { resolve_command_on_shutdown(cmd); },
                   },
                   *remaining);
    }

    request_slots_->fail_all(error);
}

void AgentRuntime::resolve_command_on_shutdown(Command& cmd) {
    auto shutdown_error = []() {
        return std::unexpected(Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    };
    std::visit(overloaded{
                   [&](SetSystemPromptCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](GetHistoryCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](ClearHistoryCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](AddSystemMessageCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](RegisterToolCmd& c) { c.done->set_value(shutdown_error()); },
                   [&](RegisterToolsCmd& c) { c.done->set_value(shutdown_error()); },
               },
               cmd);
}

bool AgentRuntime::refresh_tool_calling_state() {
    auto metadata = tool_registry_.get_all_tool_metadata();
    std::vector<CoreToolInfo> tools;
    tools.reserve(metadata.size());
    for (const auto& tm : metadata) {
        tools.push_back(CoreToolInfo{
            tm.name,
            tm.description,
            tm.parameters_schema.dump(),
        });
    }

    bool active = false;
    if (tools.empty()) {
        backend_->clear_tool_grammar();
    } else if (backend_->set_tool_calling(tools)) {
        active = true;
        ZOO_LOG("info", "tool calling configured (%zu tools, format=%s)", tools.size(),
                backend_->tool_calling_format_name());
    } else {
        backend_->clear_tool_grammar();
        ZOO_LOG("warn", "tool calling setup failed, falling back to unconstrained generation");
    }
    tool_grammar_active_.store(active, std::memory_order_release);
    return active;
}

void AgentRuntime::enforce_history_limit() {
    backend_->trim_history(agent_config_.max_history_messages);
}

GenerationOptions
AgentRuntime::resolve_generation_options(const GenerationOptions& overrides) const {
    if (overrides.is_default()) {
        return default_generation_options_;
    }
    return overrides;
}

template <typename Result>
RequestHandle<Result> AgentRuntime::make_immediate_error_handle(Error error) {
    auto state = std::make_shared<ImmediateRequestState<Result>>(
        Expected<Result>(std::unexpected(std::move(error))));
    return RequestHandle<Result>{std::move(state), 0};
}

template <typename Result>
RequestHandle<Result> AgentRuntime::enqueue_request(RequestPayload payload) {
    if (!running_.load(std::memory_order_acquire)) {
        return make_immediate_error_handle<Result>(
            Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }

    if (auto validation = payload.options.validate(); !validation) {
        return make_immediate_error_handle<Result>(validation.error());
    }

    auto reservation = request_slots_->emplace(std::move(payload));
    if (!reservation) {
        return make_immediate_error_handle<Result>(reservation.error());
    }

    auto state = std::make_shared<SlotRequestState<Result>>(request_slots_, reservation->slot,
                                                            reservation->generation);
    RequestHandle<Result> handle{std::move(state), reservation->id};

    if (!request_mailbox_.push_request(QueuedRequest{reservation->slot, reservation->generation})) {
        request_slots_->resolve_error(reservation->slot, reservation->generation,
                                      Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }

    return handle;
}

template RequestHandle<TextResponse> AgentRuntime::make_immediate_error_handle(Error error);
template RequestHandle<ExtractionResponse> AgentRuntime::make_immediate_error_handle(Error error);
template RequestHandle<TextResponse> AgentRuntime::enqueue_request(RequestPayload payload);
template RequestHandle<ExtractionResponse> AgentRuntime::enqueue_request(RequestPayload payload);

} // namespace zoo::internal::agent
