/**
 * @file runtime_tool_loop.cpp
 * @brief Agentic tool loop: generate → detect tool calls → execute → re-generate.
 */

#include "zoo/internal/agent/runtime.hpp"

#include "zoo/internal/agent/runtime_helpers.hpp"
#include "zoo/internal/log.hpp"
#include "zoo/tools/validation.hpp"
#include <chrono>
#include <unordered_map>

namespace zoo::internal::agent {

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
    const bool use_native_tool_calling =
        has_tools && tool_grammar_active_.load(std::memory_order_acquire);

    ZOO_LOG("debug", "processing request %lu (tools=%d, native_tc=%d)",
            static_cast<unsigned long>(request.id), has_tools, use_native_tool_calling);

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

        if (request.streaming_callback) {
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
        std::vector<ToolCallInfo> structured_tool_calls;

        if (use_native_tool_calling && generated->tool_call_detected) {
            // Parse the output using the model's native format parser.
            auto parsed = backend_->parse_tool_response(generated->text);
            response_text = std::move(parsed.content);

            if (!parsed.tool_calls.empty()) {
                structured_tool_calls = std::move(parsed.tool_calls);
                const auto& first_tc = structured_tool_calls.front();
                tools::ToolCall tc;
                tc.id = first_tc.id;
                tc.name = first_tc.name;
                try {
                    tc.arguments = nlohmann::json::parse(first_tc.arguments_json);
                } catch (const nlohmann::json::exception&) {
                    tc.arguments = nlohmann::json::object();
                }
                detected_tool_call = std::move(tc);
            }
        } else {
            response_text = std::move(generated->text);
        }

        if (detected_tool_call.has_value()) {
            const auto& tool_call = *detected_tool_call;

            // Store assistant message with structured tool calls for proper
            // template rendering in subsequent turns.
            if (!structured_tool_calls.empty()) {
                backend_->add_message(
                    Message::assistant_with_tool_calls(response_text, structured_tool_calls));
            } else {
                backend_->add_message(Message::assistant(response_text));
            }
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

            ZOO_LOG("info", "invoking tool '%s' (iteration %d, native_tc=%d)",
                    tool_call.name.c_str(), iteration, use_native_tool_calling);
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

} // namespace zoo::internal::agent
