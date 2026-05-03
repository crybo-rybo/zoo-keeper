/**
 * @file runtime.cpp
 * @brief Public request submission methods for the internal agent runtime.
 */

#include "agent/runtime.hpp"

#include "agent/request_state.hpp"
#include "zoo/tools/validation.hpp"
#include <utility>
#include <vector>

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

} // namespace

RequestHandle<TextResponse> AgentRuntime::chat(std::string_view user_message,
                                               const GenerationOptions& options,
                                               AsyncTokenCallback callback) {
    return chat(MessageView{Role::User, user_message}, options, std::move(callback));
}

RequestHandle<TextResponse> AgentRuntime::chat(MessageView message,
                                               const GenerationOptions& options,
                                               AsyncTokenCallback callback) {
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
                                                   AsyncTokenCallback callback) {
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
                                                        AsyncTokenCallback callback) {
    return extract(output_schema, MessageView{Role::User, user_message}, options,
                   std::move(callback));
}

RequestHandle<ExtractionResponse> AgentRuntime::extract(const nlohmann::json& output_schema,
                                                        MessageView message,
                                                        const GenerationOptions& options,
                                                        AsyncTokenCallback callback) {
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
                                                        AsyncTokenCallback callback) {
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
RequestHandle<Result> AgentRuntime::enqueue_request(RequestPayload&& payload) {
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

    auto state = std::make_shared<SlotRequestState<Result>>(
        request_slots_, reservation->id, reservation->slot, reservation->generation);
    RequestHandle<Result> handle{std::move(state), reservation->id};

    if (!request_mailbox_.push_request(QueuedRequest{reservation->slot, reservation->generation})) {
        request_slots_->resolve_error(reservation->slot, reservation->generation,
                                      Error{ErrorCode::AgentNotRunning, "Agent is not running"});
    }

    return handle;
}

template RequestHandle<TextResponse> AgentRuntime::make_immediate_error_handle(Error error);
template RequestHandle<ExtractionResponse> AgentRuntime::make_immediate_error_handle(Error error);
template RequestHandle<TextResponse> AgentRuntime::enqueue_request(RequestPayload&& payload);
template RequestHandle<ExtractionResponse> AgentRuntime::enqueue_request(RequestPayload&& payload);

} // namespace zoo::internal::agent
