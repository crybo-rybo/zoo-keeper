/**
 * @file runtime_extraction.cpp
 * @brief Structured output extraction: schema-constrained single-pass generation.
 */

#include "zoo/internal/agent/runtime.hpp"

#include "zoo/core/model.hpp"
#include "zoo/internal/agent/runtime_helpers.hpp"
#include "zoo/internal/log.hpp"
#include "zoo/internal/tools/grammar.hpp"
#include "zoo/tools/registry.hpp"
#include "zoo/tools/validation.hpp"
#include <chrono>

namespace zoo::internal::agent {

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

    // Set schema grammar, restore previous tool calling state on exit
    const bool had_tool_calling = tool_grammar_active_.load(std::memory_order_acquire);

    if (!backend_->set_schema_grammar(grammar_str)) {
        return std::unexpected(
            Error{ErrorCode::ExtractionFailed, "Failed to initialize schema grammar"});
    }

    ScopeExit grammar_guard([this, had_tool_calling] {
        backend_->clear_tool_grammar();
        if (had_tool_calling) {
            // Restore native tool calling by re-sending tool metadata.
            auto metadata = tool_registry_.get_all_tool_metadata();
            std::vector<CoreToolInfo> tools;
            tools.reserve(metadata.size());
            for (const auto& tm : metadata) {
                tools.push_back(CoreToolInfo{tm.name, tm.description, tm.parameters_schema.dump()});
            }
            bool restored = backend_->set_tool_calling(tools);
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

} // namespace zoo::internal::agent
