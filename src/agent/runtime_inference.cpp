/**
 * @file runtime_inference.cpp
 * @brief Inference-thread dispatch and the agentic tool loop.
 */

#include "agent/runtime.hpp"

#include "agent/cancellation.hpp"
#include "agent/runtime_helpers.hpp"
#include "log.hpp"
#include "zoo/tools/validation.hpp"
#include <chrono>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace zoo::internal::agent {

namespace {

class ToolLoopController {
  public:
    ToolLoopController(AgentBackend& backend, const tools::ToolRegistry& tool_registry,
                       ToolExecutor& tool_executor, CallbackDispatcher& callback_dispatcher,
                       const CancellationToken& stop_token, const AgentConfig& agent_config,
                       bool use_native_tool_calling)
        : backend_(backend), tool_registry_(tool_registry), tool_executor_(tool_executor),
          callback_dispatcher_(callback_dispatcher), stop_token_(stop_token),
          agent_config_(agent_config), use_native_tool_calling_(use_native_tool_calling) {}

    Expected<TextResponse> run(const ActiveRequest& request,
                               std::chrono::steady_clock::time_point start_time) {
        GenerationStats stats(start_time);
        GenerationRunner generation_runner(backend_, callback_dispatcher_);
        CompositeCancellation cancel(stop_token_, request.cancelled);

        for (int iteration = 1; iteration <= agent_config_.max_tool_iterations; ++iteration) {
            if (cancel.cancelled()) {
                return std::unexpected(
                    Error{ErrorCode::RequestCancelled, "Request cancelled during tool loop"});
            }

            auto cancellation_check = [&cancel]() { return cancel.cancelled(); };
            auto pass = generation_runner.run(*request.options, request.streaming_callback,
                                              CancellationCallback(cancellation_check), stats);
            if (!pass) {
                return std::unexpected(pass.error());
            }

            ToolDetection detection = detect_tool_calls(std::move(pass->generation));
            if (!detection.structured_tool_calls.empty()) {
                auto tool_result = execute_turn_tool_calls(detection, iteration, cancel,
                                                           request.options->record_tool_trace);
                if (!tool_result) {
                    return std::unexpected(tool_result.error());
                }
                callback_dispatcher_.drain();
                continue;
            }

            if (detection.response_text.empty() && any_tool_invoked_ &&
                iteration < agent_config_.max_tool_iterations) {
                backend_.add_message(
                    Message::user("Please respond to the user with the tool result.").view());
                callback_dispatcher_.drain();
                continue;
            }

            return finish_response(std::move(detection.response_text), stats,
                                   request.options->record_tool_trace);
        }

        ZOO_LOG("error", "tool loop iteration limit reached (%d)",
                agent_config_.max_tool_iterations);
        return std::unexpected(Error{ErrorCode::ToolLoopLimitReached,
                                     "Tool loop iteration limit reached (" +
                                         std::to_string(agent_config_.max_tool_iterations) + ")"});
    }

  private:
    struct ToolDetection {
        std::string response_text;
        std::vector<ToolCallInfo> structured_tool_calls;
    };

    ToolDetection detect_tool_calls(GenerationResult generated) const {
        ToolDetection detection;
        if (!use_native_tool_calling_ || !generated.tool_call_detected) {
            detection.response_text = std::move(generated.text);
            return detection;
        }

        if (!generated.tool_calls.empty()) {
            detection.response_text = std::move(generated.parsed_content);
            detection.structured_tool_calls = std::move(generated.tool_calls);
        } else {
            auto parsed = backend_.parse_tool_response(generated.text);
            detection.response_text = std::move(parsed.content);
            detection.structured_tool_calls = std::move(parsed.tool_calls);
        }
        return detection;
    }

    static tools::ToolCall to_runtime_tool_call(const ToolCallInfo& info) {
        tools::ToolCall tool_call;
        tool_call.id = info.id;
        tool_call.name = info.name;
        try {
            tool_call.arguments = nlohmann::json::parse(info.arguments_json);
        } catch (const nlohmann::json::exception&) {
            tool_call.arguments = nlohmann::json::object();
        }
        return tool_call;
    }

    /// Executes every tool call emitted by a single assistant turn. The
    /// assistant message (with all tool calls attached) is appended once; each
    /// tool call then produces its own Tool message with the result. Validation
    /// failures on one call do not abort sibling calls within the same turn.
    Expected<void> execute_turn_tool_calls(const ToolDetection& detection, int iteration,
                                           const CompositeCancellation& cancel,
                                           bool record_tool_trace) {
        backend_.add_message(Message::assistant_with_tool_calls(detection.response_text,
                                                                detection.structured_tool_calls)
                                 .view());
        backend_.finalize_response();

        for (const auto& tc_info : detection.structured_tool_calls) {
            if (cancel.cancelled()) {
                return std::unexpected(Error{ErrorCode::RequestCancelled,
                                             "Request cancelled before tool call could run"});
            }

            const tools::ToolCall tool_call = to_runtime_tool_call(tc_info);
            std::string args_json = tc_info.arguments_json;

            if (auto validation = validator_.validate(tool_call, tool_registry_); !validation) {
                if (auto fail = handle_validation_failure(tool_call, std::move(args_json),
                                                          validation.error(), record_tool_trace);
                    !fail) {
                    return std::unexpected(fail.error());
                }
                continue;
            }

            if (auto invoke = invoke_tool_handler(tool_call, std::move(args_json), iteration,
                                                  cancel, record_tool_trace);
                !invoke) {
                return std::unexpected(invoke.error());
            }
        }

        return {};
    }

    Expected<void> invoke_tool_handler(const tools::ToolCall& tool_call, std::string args_json,
                                       int iteration, const CompositeCancellation& cancel,
                                       bool record_tool_trace) {
        ZOO_LOG("info", "invoking tool '%s' (iteration %d, native_tc=%d)", tool_call.name.c_str(),
                iteration, use_native_tool_calling_);

        auto handler = tool_registry_.find_handler(tool_call.name);
        Expected<nlohmann::json> invoke_result;
        if (handler) {
            auto future = tool_executor_.submit(std::move(*handler), tool_call.arguments);
            invoke_result = ToolExecutor::wait_for_result(future, cancel);
        } else {
            invoke_result = std::unexpected(
                Error{ErrorCode::ToolNotFound, "Tool not found: " + tool_call.name});
        }

        if (!invoke_result && invoke_result.error().code == ErrorCode::RequestCancelled) {
            return std::unexpected(invoke_result.error());
        }

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

        backend_.add_message(Message::tool(std::move(tool_result_str), tool_call.id).view());
        any_tool_invoked_ = true;
        if (record_tool_trace) {
            tool_invocations_.push_back(
                ToolInvocation{tool_call.id, tool_call.name, std::move(args_json), status,
                               std::move(result_json), std::move(tool_error)});
        }
        return {};
    }

    Expected<void> handle_validation_failure(const tools::ToolCall& tool_call,
                                             std::string args_json, Error validation_error,
                                             bool record_tool_trace) {
        int& retry_count = retry_count_for(tool_call.name);
        if (retry_count >= agent_config_.max_tool_retries) {
            ZOO_LOG("error", "tool retries exhausted for '%s': %s", tool_call.name.c_str(),
                    validation_error.message.c_str());
            return std::unexpected(Error{ErrorCode::ToolRetriesExhausted,
                                         "Tool retries exhausted for '" + tool_call.name +
                                             "': " + validation_error.message});
        }

        ++retry_count;
        ZOO_LOG("warn", "tool '%s' validation failed (retry %d/%d): %s", tool_call.name.c_str(),
                retry_count, agent_config_.max_tool_retries, validation_error.message.c_str());

        std::string error_content = "Error: " + validation_error.message;
        backend_.add_message(
            Message::tool(error_content + "\nPlease correct the arguments.", tool_call.id).view());
        any_tool_invoked_ = true;
        if (record_tool_trace) {
            tool_invocations_.push_back(ToolInvocation{
                tool_call.id, tool_call.name, std::move(args_json),
                ToolInvocationStatus::ValidationFailed, std::nullopt, std::move(validation_error)});
        }
        return {};
    }

    int& retry_count_for(std::string_view tool_name) {
        for (auto& entry : retry_counts_) {
            if (entry.first == tool_name) {
                return entry.second;
            }
        }
        retry_counts_.emplace_back(std::string(tool_name), 0);
        return retry_counts_.back().second;
    }

    Expected<TextResponse> finish_response(std::string response_text, const GenerationStats& stats,
                                           bool record_tool_trace) {
        const auto end_time = std::chrono::steady_clock::now();

        backend_.add_message(Message::assistant(response_text).view());
        backend_.finalize_response();
        callback_dispatcher_.drain();

        TextResponse response;
        response.text = std::move(response_text);
        if (record_tool_trace && !tool_invocations_.empty()) {
            response.tool_trace = ToolTrace{std::move(tool_invocations_)};
        }
        response.usage = stats.usage();
        response.metrics = stats.metrics(end_time);
        return response;
    }

    AgentBackend& backend_;
    const tools::ToolRegistry& tool_registry_;
    ToolExecutor& tool_executor_;
    CallbackDispatcher& callback_dispatcher_;
    const CancellationToken& stop_token_;
    const AgentConfig& agent_config_;
    bool use_native_tool_calling_;
    tools::ToolArgumentsValidator validator_;
    bool any_tool_invoked_ = false;
    std::vector<std::pair<std::string, int>> retry_counts_;
    std::vector<ToolInvocation> tool_invocations_;
};

} // namespace

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

namespace {

/// Drains the callback dispatcher, swallowing any captured exception.  The
/// inference thread invokes this before resolving the slot so that no
/// dispatcher entry can outlive the slot's streaming callback (which is owned
/// by the request payload and is destroyed when the awaiting thread releases
/// the slot).  Streaming-callback exceptions are converted into a runtime
/// error by the caller when appropriate.
[[nodiscard]] std::optional<Error>
drain_dispatcher_swallowing_errors(CallbackDispatcher& dispatcher) noexcept {
    try {
        dispatcher.drain();
        return std::nullopt;
    } catch (const std::exception& e) {
        return Error{ErrorCode::InferenceFailed,
                     std::string("Streaming callback threw: ") + e.what()};
    } catch (...) {
        return Error{ErrorCode::InferenceFailed, "Streaming callback threw unknown exception"};
    }
}

} // namespace

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

    const ResultKind result_kind = active_request->result_kind;
    if (result_kind == ResultKind::Extraction) {
        Expected<ExtractionResponse> result =
            std::unexpected(Error{ErrorCode::InferenceFailed, "Inference did not run"});
        try {
            result = process_extraction_request(*active_request);
        } catch (const std::exception& e) {
            ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
            result = std::unexpected(
                Error{ErrorCode::InferenceFailed, std::string("Unhandled exception: ") + e.what()});
        } catch (...) {
            ZOO_LOG("error", "unknown exception in inference thread");
            result = std::unexpected(
                Error{ErrorCode::InferenceFailed, "Unknown exception in inference thread"});
        }
        // CRITICAL: drain the dispatcher before resolving the slot — see note
        // below the text-result branch.
        if (auto drain_error = drain_dispatcher_swallowing_errors(callback_dispatcher_)) {
            if (result.has_value()) {
                result = std::unexpected(std::move(*drain_error));
            }
        }
        request_slots_->resolve_extraction(request.slot, request.generation, std::move(result));
        return;
    }

    Expected<TextResponse> result =
        std::unexpected(Error{ErrorCode::InferenceFailed, "Inference did not run"});
    try {
        result = process_request(*active_request);
    } catch (const std::exception& e) {
        ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
        result = std::unexpected(
            Error{ErrorCode::InferenceFailed, std::string("Unhandled exception: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "unknown exception in inference thread");
        result = std::unexpected(
            Error{ErrorCode::InferenceFailed, "Unknown exception in inference thread"});
    }

    // CRITICAL: drain the dispatcher before resolving the slot. Streaming
    // entries hold a raw pointer to the request's AsyncTokenCallback, which
    // lives inside slot.payload. Once we resolve the slot, the awaiting thread
    // may call await/release at any moment and destroy that payload — any
    // dispatcher entry still pending at that point would be a use-after-free.
    if (auto drain_error = drain_dispatcher_swallowing_errors(callback_dispatcher_)) {
        if (result.has_value()) {
            result = std::unexpected(std::move(*drain_error));
        }
    }
    request_slots_->resolve_text(request.slot, request.generation, std::move(result));
}

Expected<TextResponse> AgentRuntime::process_request(const ActiveRequest& request) {
    auto start_time = std::chrono::steady_clock::now();

    auto history_scope =
        RequestHistoryScope::enter(*backend_, request.history_mode, *request.messages,
                                   agent_config_.max_history_messages, "chat");
    if (!history_scope) {
        return std::unexpected(history_scope.error());
    }

    const bool has_tools = tool_registry_.size() > 0;
    const bool use_native_tool_calling =
        has_tools && tool_grammar_active_.load(std::memory_order_acquire);

    ZOO_LOG("debug", "processing request %lu (tools=%d, native_tc=%d)",
            static_cast<unsigned long>(request.id), has_tools, use_native_tool_calling);

    ToolLoopController tool_loop(*backend_, tool_registry_, tool_executor_, callback_dispatcher_,
                                 stop_token_, agent_config_, use_native_tool_calling);
    return tool_loop.run(request, start_time);
}

} // namespace zoo::internal::agent
