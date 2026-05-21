/**
 * @file runtime_inference.cpp
 * @brief Inference-thread dispatch and the agentic tool loop.
 */

#include "agent/runtime.hpp"

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
                       const AgentConfig& agent_config, bool use_native_tool_calling)
        : backend_(backend), tool_registry_(tool_registry), tool_executor_(tool_executor),
          callback_dispatcher_(callback_dispatcher), agent_config_(agent_config),
          use_native_tool_calling_(use_native_tool_calling) {}

    Expected<TextResponse> run(const ActiveRequest& request,
                               std::chrono::steady_clock::time_point start_time) {
        GenerationStats stats(start_time);
        GenerationRunner generation_runner(backend_, callback_dispatcher_);

        for (int iteration = 1; iteration <= agent_config_.max_tool_iterations; ++iteration) {
            if (is_cancelled(request)) {
                return std::unexpected(
                    Error{ErrorCode::RequestCancelled, "Request cancelled during tool loop"});
            }

            auto cancellation_check = [&request]() { return is_cancelled(request); };
            auto pass = generation_runner.run(*request.options, request.streaming_callback,
                                              CancellationCallback(cancellation_check), stats);
            if (!pass) {
                return std::unexpected(pass.error());
            }

            ToolDetection detection = detect_tool_call(std::move(pass->generation));
            if (detection.tool_call.has_value()) {
                auto tool_result =
                    handle_tool_call(*detection.tool_call, std::move(detection.response_text),
                                     std::move(detection.structured_tool_calls), iteration,
                                     request.options->record_tool_trace);
                if (!tool_result) {
                    return std::unexpected(tool_result.error());
                }
                continue;
            }

            if (detection.response_text.empty() && tool_invoked_ &&
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
        std::optional<tools::ToolCall> tool_call;
        std::string response_text;
        std::vector<ToolCallInfo> structured_tool_calls;
    };

    [[nodiscard]] static bool is_cancelled(const ActiveRequest& request) {
        return request.cancelled && request.cancelled->load(std::memory_order_acquire);
    }

    ToolDetection detect_tool_call(GenerationResult generated) const {
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

        if (detection.structured_tool_calls.empty()) {
            return detection;
        }

        const auto& first_tc = detection.structured_tool_calls.front();
        tools::ToolCall tool_call;
        tool_call.id = first_tc.id;
        tool_call.name = first_tc.name;
        try {
            tool_call.arguments = nlohmann::json::parse(first_tc.arguments_json);
        } catch (const nlohmann::json::exception&) {
            tool_call.arguments = nlohmann::json::object();
        }
        detection.tool_call = std::move(tool_call);
        return detection;
    }

    Expected<void> handle_tool_call(const tools::ToolCall& tool_call, std::string response_text,
                                    std::vector<ToolCallInfo> structured_tool_calls, int iteration,
                                    bool record_tool_trace) {
        if (!structured_tool_calls.empty()) {
            backend_.add_message(
                Message::assistant_with_tool_calls(response_text, structured_tool_calls).view());
        } else {
            backend_.add_message(Message::assistant(response_text).view());
        }
        backend_.finalize_response();

        std::string args_json = structured_tool_calls.empty()
                                    ? tool_call.arguments.dump()
                                    : structured_tool_calls.front().arguments_json;
        if (auto validation_result = validator_.validate(tool_call, tool_registry_);
            !validation_result) {
            return handle_validation_failure(tool_call, std::move(args_json),
                                             validation_result.error(), record_tool_trace);
        }

        ZOO_LOG("info", "invoking tool '%s' (iteration %d, native_tc=%d)", tool_call.name.c_str(),
                iteration, use_native_tool_calling_);
        auto handler = tool_registry_.find_handler(tool_call.name);
        Expected<nlohmann::json> invoke_result =
            handler ? tool_executor_.submit(std::move(*handler), tool_call.arguments).get()
                    : std::unexpected(
                          Error{ErrorCode::ToolNotFound, "Tool not found: " + tool_call.name});

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
        tool_invoked_ = true;
        if (record_tool_trace) {
            tool_invocations_.push_back(
                ToolInvocation{tool_call.id, tool_call.name, std::move(args_json), status,
                               std::move(result_json), std::move(tool_error)});
        }
        callback_dispatcher_.drain();
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
        tool_invoked_ = true;
        if (record_tool_trace) {
            tool_invocations_.push_back(ToolInvocation{
                tool_call.id, tool_call.name, std::move(args_json),
                ToolInvocationStatus::ValidationFailed, std::nullopt, std::move(validation_error)});
        }
        callback_dispatcher_.drain();
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
    const AgentConfig& agent_config_;
    bool use_native_tool_calling_;
    tools::ToolArgumentsValidator validator_;
    bool tool_invoked_ = false;
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
                                 agent_config_, use_native_tool_calling);
    return tool_loop.run(request, start_time);
}

} // namespace zoo::internal::agent
