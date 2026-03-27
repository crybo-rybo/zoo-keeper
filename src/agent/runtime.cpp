/**
 * @file runtime.cpp
 * @brief Internal worker-owned runtime: lifecycle, request submission, and dispatch.
 *
 * The agentic tool loop lives in runtime_tool_loop.cpp.
 * Structured output extraction lives in runtime_extraction.cpp.
 */

#include "zoo/internal/agent/runtime.hpp"

#include "zoo/core/model.hpp"
#include "zoo/internal/log.hpp"
#include "zoo/tools/registry.hpp"
#include <cassert>
#include <thread>

namespace zoo::internal::agent {

namespace {

template <typename Result> struct ImmediateResultState {
    std::optional<Expected<Result>> result;
};

template <typename Result>
Expected<Result> await_immediate_handle(void* state, uint32_t, uint32_t) {
    auto& immediate = *static_cast<ImmediateResultState<Result>*>(state);
    if (!immediate.result.has_value()) {
        return std::unexpected(
            Error{ErrorCode::AgentNotRunning, "Request result is no longer available"});
    }
    auto result = std::move(*immediate.result);
    immediate.result.reset();
    return result;
}

template <typename Result> bool ready_immediate_handle(const void*, uint32_t, uint32_t) {
    return true;
}

template <typename Result> void release_immediate_handle(void* state, uint32_t, uint32_t) {
    auto& immediate = *static_cast<ImmediateResultState<Result>*>(state);
    immediate.result.reset();
}

std::vector<Message> materialize_conversation(ConversationView messages) {
    std::vector<Message> owned;
    owned.reserve(messages.size());
    for (size_t index = 0; index < messages.size(); ++index) {
        owned.push_back(Message::from_view(messages[index]));
    }
    return owned;
}

} // namespace

AgentRuntime::AgentRuntime(ModelConfig model_config, AgentConfig agent_config,
                           GenerationOptions default_generation,
                           std::unique_ptr<AgentBackend> backend)
    : model_config_(std::move(model_config)), agent_config_(agent_config),
      default_generation_options_(std::move(default_generation)), backend_(std::move(backend)),
      request_slots_(std::make_shared<RequestSlots>(agent_config_.request_queue_capacity)),
      request_mailbox_(agent_config_.request_queue_capacity) {
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

void AgentRuntime::set_system_prompt(std::string_view prompt) {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    auto done = std::make_shared<std::promise<void>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(SetSystemPromptCmd{std::string(prompt), std::move(done)})) {
        return;
    }
    future.get();
}

HistorySnapshot AgentRuntime::get_history() const {
    if (!running_.load(std::memory_order_acquire)) {
        return {};
    }

    auto done = std::make_shared<std::promise<HistorySnapshot>>();
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

    update_tool_calling();
    return {};
}

Expected<void> AgentRuntime::register_tools(std::vector<tools::ToolDefinition> definitions) {
    assert(!inference_thread_.joinable() ||
           std::this_thread::get_id() != inference_thread_.get_id());

    if (definitions.empty()) {
        return {};
    }

    if (auto result = tool_registry_.register_tools(std::move(definitions)); !result) {
        return std::unexpected(result.error());
    }

    update_tool_calling();
    return {};
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
                   [this](RefreshToolCallingCmd& c) {
                       bool active = false;
                       if (c.tools.empty()) {
                           backend_->clear_tool_grammar();
                       } else if (backend_->set_tool_calling(c.tools)) {
                           active = true;
                           ZOO_LOG("info", "tool calling configured (%zu tools, format=%s)",
                                   c.tools.size(), backend_->tool_calling_format_name());
                       } else {
                           backend_->clear_tool_grammar();
                           ZOO_LOG("warn", "tool calling setup failed, falling back to "
                                           "unconstrained generation");
                       }
                       tool_grammar_active_.store(active, std::memory_order_release);
                       c.done->set_value(active);
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
    std::visit(overloaded{
                   [](SetSystemPromptCmd& c) { c.done->set_value(); },
                   [](GetHistoryCmd& c) { c.done->set_value(HistorySnapshot{}); },
                   [](ClearHistoryCmd& c) { c.done->set_value(); },
                   [](RefreshToolCallingCmd& c) { c.done->set_value(false); },
               },
               cmd);
}

void AgentRuntime::update_tool_calling() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

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

    auto done = std::make_shared<std::promise<bool>>();
    auto future = done->get_future();
    if (!request_mailbox_.push_command(RefreshToolCallingCmd{std::move(tools), std::move(done)})) {
        return;
    }
    future.get();
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
    auto state = std::make_shared<ImmediateResultState<Result>>();
    state->result = std::unexpected(std::move(error));
    return RequestHandle<Result>{0,
                                 std::move(state),
                                 0,
                                 0,
                                 &await_immediate_handle<Result>,
                                 &ready_immediate_handle<Result>,
                                 &release_immediate_handle<Result>};
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

    RequestHandle<Result> handle;
    if constexpr (std::same_as<Result, TextResponse>) {
        handle = RequestHandle<Result>{reservation->id,
                                       request_slots_,
                                       reservation->slot,
                                       reservation->generation,
                                       &RequestSlots::await_text_handle,
                                       &RequestSlots::ready_handle,
                                       &RequestSlots::release_handle};
    } else {
        handle = RequestHandle<Result>{reservation->id,
                                       request_slots_,
                                       reservation->slot,
                                       reservation->generation,
                                       &RequestSlots::await_extraction_handle,
                                       &RequestSlots::ready_handle,
                                       &RequestSlots::release_handle};
    }

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
