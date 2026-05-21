/**
 * @file runtime_inference.cpp
 * @brief Inference-thread dispatch and the agentic tool loop.
 */

#include "agent/runtime.hpp"

#include "agent/runtime_helpers.hpp"
#include "log.hpp"
#include "zoo/tools/validation.hpp"
#include <atomic>
#include <chrono>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace zoo::internal::agent {

namespace {

[[nodiscard]] bool is_cancelled(const std::atomic<bool>& running, const ActiveRequest& request) {
    return !running.load(std::memory_order_acquire) ||
           (request.cancelled && request.cancelled->load(std::memory_order_acquire));
}

/// Drains the callback dispatcher and, if drain throws, folds the captured
/// streaming-callback exception into `result` (only when `result` was a
/// success, so we don't shadow a more informative upstream error).
template <typename T>
void fold_dispatcher_drain_into(CallbackDispatcher& dispatcher, Expected<T>& result) {
    try {
        dispatcher.drain();
    } catch (const std::exception& e) {
        if (result.has_value()) {
            result = std::unexpected(Error{ErrorCode::InferenceFailed,
                                           std::string("Streaming callback threw: ") + e.what()});
        }
    } catch (...) {
        if (result.has_value()) {
            result = std::unexpected(
                Error{ErrorCode::InferenceFailed, "Streaming callback threw unknown exception"});
        }
    }
}

class ToolLoopController {
  public:
    ToolLoopController(AgentBackend& backend, const tools::ToolRegistry& tool_registry,
                       ToolExecutor& tool_executor, CallbackDispatcher& callback_dispatcher,
                       const std::atomic<bool>& running, const AgentConfig& agent_config,
                       bool use_native_tool_calling)
        : backend_(backend), tool_registry_(tool_registry), tool_executor_(tool_executor),
          callback_dispatcher_(callback_dispatcher), running_(running), agent_config_(agent_config),
          use_native_tool_calling_(use_native_tool_calling) {}

    Expected<TextResponse> run(const ActiveRequest& request,
                               std::chrono::steady_clock::time_point start_time) {
        GenerationStats stats(start_time);
        GenerationRunner generation_runner(backend_, callback_dispatcher_);
        auto cancellation_check = [&] { return is_cancelled(running_, request); };

        for (int iteration = 1; iteration <= agent_config_.max_tool_iterations; ++iteration) {
            if (cancellation_check()) {
                return std::unexpected(
                    Error{ErrorCode::RequestCancelled, "Request cancelled during tool loop"});
            }

            auto pass = generation_runner.run(*request.options, request.streaming_callback,
                                              CancellationCallback(cancellation_check), stats);
            if (!pass) {
                return std::unexpected(pass.error());
            }

            auto [response_text, tool_calls] = detect_tool_calls(std::move(pass->generation));

            if (!tool_calls.empty()) {
                auto tool_result =
                    execute_tool_calls(response_text, tool_calls, request, iteration);
                if (!tool_result) {
                    return std::unexpected(tool_result.error());
                }
                callback_dispatcher_.drain();
                continue;
            }

            if (response_text.empty() && any_tool_invoked_ &&
                iteration < agent_config_.max_tool_iterations) {
                backend_.add_message(
                    Message::user("Please respond to the user with the tool result.").view());
                callback_dispatcher_.drain();
                continue;
            }

            return finish_response(std::move(response_text), stats,
                                   request.options->record_tool_trace);
        }

        ZOO_LOG("error", "tool loop iteration limit reached (%d)",
                agent_config_.max_tool_iterations);
        return std::unexpected(Error{ErrorCode::ToolLoopLimitReached,
                                     "Tool loop iteration limit reached (" +
                                         std::to_string(agent_config_.max_tool_iterations) + ")"});
    }

  private:
    struct DetectionResult {
        std::string response_text;
        std::vector<ToolCallInfo> tool_calls;
    };

    DetectionResult detect_tool_calls(GenerationResult generated) const {
        if (!use_native_tool_calling_ || !generated.tool_call_detected) {
            return {std::move(generated.text), {}};
        }
        if (!generated.tool_calls.empty()) {
            return {std::move(generated.parsed_content), std::move(generated.tool_calls)};
        }
        auto parsed = backend_.parse_tool_response(generated.text);
        return {std::move(parsed.content), std::move(parsed.tool_calls)};
    }

    /// Executes every tool call from a single assistant turn. The assistant
    /// message is appended once with all tool calls attached; one Tool message
    /// is appended per call. A validation failure on one call does NOT abort
    /// sibling calls within the same turn.
    Expected<void> execute_tool_calls(const std::string& response_text,
                                      const std::vector<ToolCallInfo>& tool_calls,
                                      const ActiveRequest& request, int iteration) {
        backend_.add_message(Message::assistant_with_tool_calls(response_text, tool_calls).view());
        backend_.finalize_response();

        const bool record_trace = request.options->record_tool_trace;
        for (const auto& info : tool_calls) {
            if (is_cancelled(running_, request)) {
                return std::unexpected(Error{ErrorCode::RequestCancelled,
                                             "Request cancelled before tool call could run"});
            }

            tools::ToolCall tool_call{info.id, info.name, nlohmann::json::object()};
            try {
                tool_call.arguments = nlohmann::json::parse(info.arguments_json);
            } catch (const nlohmann::json::exception&) {
                // arguments remain an empty object; validation will reject.
            }

            if (auto validation = validator_.validate(tool_call, tool_registry_); !validation) {
                if (auto r = handle_validation_failure(tool_call, info.arguments_json,
                                                       validation.error(), record_trace);
                    !r) {
                    return std::unexpected(r.error());
                }
                continue;
            }

            if (auto r =
                    invoke_tool(tool_call, info.arguments_json, iteration, request, record_trace);
                !r) {
                return std::unexpected(r.error());
            }
        }
        return {};
    }

    Expected<void> invoke_tool(const tools::ToolCall& tool_call, std::string args_json,
                               int iteration, const ActiveRequest& request, bool record_trace) {
        ZOO_LOG("info", "invoking tool '%s' (iteration %d, native_tc=%d)", tool_call.name.c_str(),
                iteration, use_native_tool_calling_);

        auto handler = tool_registry_.find_handler(tool_call.name);
        Expected<nlohmann::json> result;
        if (handler) {
            auto future = tool_executor_.submit(std::move(*handler), tool_call.arguments);
            result = ToolExecutor::wait_for_result(future, running_, request.cancelled);
        } else {
            result = std::unexpected(
                Error{ErrorCode::ToolNotFound, "Tool not found: " + tool_call.name});
        }

        if (!result && result.error().code == ErrorCode::RequestCancelled) {
            return std::unexpected(result.error());
        }

        std::string tool_msg;
        std::optional<std::string> result_json;
        std::optional<Error> tool_error;
        ToolInvocationStatus status = ToolInvocationStatus::Succeeded;
        if (result) {
            tool_msg = result->dump();
            result_json = tool_msg;
        } else {
            tool_msg = "Error: " + result.error().message;
            tool_error = result.error();
            status = ToolInvocationStatus::ExecutionFailed;
        }

        backend_.add_message(Message::tool(std::move(tool_msg), tool_call.id).view());
        any_tool_invoked_ = true;
        if (record_trace) {
            tool_invocations_.push_back(
                ToolInvocation{tool_call.id, tool_call.name, std::move(args_json), status,
                               std::move(result_json), std::move(tool_error)});
        }
        return {};
    }

    Expected<void> handle_validation_failure(const tools::ToolCall& tool_call,
                                             std::string args_json, Error validation_error,
                                             bool record_trace) {
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

        backend_.add_message(
            Message::tool("Error: " + validation_error.message + "\nPlease correct the arguments.",
                          tool_call.id)
                .view());
        any_tool_invoked_ = true;
        if (record_trace) {
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
                                           bool record_trace) {
        const auto end_time = std::chrono::steady_clock::now();
        backend_.add_message(Message::assistant(response_text).view());
        backend_.finalize_response();
        callback_dispatcher_.drain();

        TextResponse response;
        response.text = std::move(response_text);
        if (record_trace && !tool_invocations_.empty()) {
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
    const std::atomic<bool>& running_;
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

    // Drain the dispatcher BEFORE resolving the slot. Streaming callbacks
    // live inside the slot payload, which is destroyed when the awaiting
    // thread releases the slot; an entry still queued at resolve time would
    // dereference freed memory. drain() also rethrows any captured
    // streaming-callback exception, which the helper folds into the result.
    try {
        if (active_request->result_kind == ResultKind::Extraction) {
            auto result = process_extraction_request(*active_request);
            fold_dispatcher_drain_into(callback_dispatcher_, result);
            request_slots_->resolve_extraction(request.slot, request.generation, std::move(result));
        } else {
            auto result = process_request(*active_request);
            fold_dispatcher_drain_into(callback_dispatcher_, result);
            request_slots_->resolve_text(request.slot, request.generation, std::move(result));
        }
    } catch (const std::exception& e) {
        ZOO_LOG("error", "unhandled exception in inference: %s", e.what());
        try {
            callback_dispatcher_.drain();
        } catch (...) {
        }
        request_slots_->resolve_error(
            request.slot, request.generation,
            Error{ErrorCode::InferenceFailed, std::string("Unhandled exception: ") + e.what()});
    } catch (...) {
        ZOO_LOG("error", "unknown exception in inference thread");
        try {
            callback_dispatcher_.drain();
        } catch (...) {
        }
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
                                 running_, agent_config_, use_native_tool_calling);
    return tool_loop.run(request, start_time);
}

} // namespace zoo::internal::agent
